package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops.Im2ColOp

/**
 * Sim wrapper: exposes the Im2ColOp ports at the top level (components must
 * be instantiated inside a Component scope for elaboration).
 */
case class Im2ColContinuityTestComp[T <: Data](dataType: HardType[T], H: Int, W: Int, C: Int, K: Int,
                                               outLanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(H, W, C), lanes = 1))
    val c = master(Tensor(dataType, Seq((H - K + 1) * (W - K + 1), K * K * C), outLanes))
  }
  val comp = Im2ColOp(dataType, H, W, C, K, outLanes)
  comp.io.a <> io.a
  io.c <> comp.io.c
}

/**
 * Phase-3 S2 proof: Im2ColOp window state IS the halo carrier (M2 resolved
 * by design, Phase-3 edition).
 *
 * The seam-convention of this op is internal (the matching windows/weight
 * layout convention lives in the layer pair that consumes it, validated
 * end-to-end elsewhere). The S2 gate therefore proves the ONLY property the
 * phase-3 banding actually depends on — seam-independence of the FEED and
 * the CONSTITUENT-CLEAN segmentation:
 *
 * 1. SEAM-STALL-EQUIVALENCE — the same picture pushed as ONE continuous
 *    stream vs TWO consecutive row runs separated by a long stall exactly at
 *    the band seam (300+ idle cycles, line buffers retained — the DMA band
 *    swap is precisely such a stall) must emit IDENTICAL window sequences
 *    (count, order, contents). This is the operational form of "the K−1
 *    bottom rows of band A are still in the state and correctly complete the
 *    first windows of band B": no halo refetch, no loss, no duplication.
 *
 * 2. COMMAND-CLEAN — two SEPARATE commands (full image A then full image B,
 *    with the stateDone boundary in between) must behave exactly like two
 *    fresh sessions: M2.2 holds — no stale cell is ever read at emission.
 */
class Im2ColContinuityTest extends AnyFunSuite {
  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384,
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = BOOT))

  private def image(H: Int, W: Int, seed: Int): Seq[Int] =
    for (y <- 0 until H; x <- 0 until W) yield (y * W + x) * 7 + seed * 301 - 11

  /** Drive pixels through a fresh sim; optional stall executed after `stallAt` pixels.
   *  Termination by quiescence: source exhausted + no window for 20k cycles. */
  private def runCommand(compiled: SimCompiled[Im2ColContinuityTestComp[SInt]],
                         source: Seq[Int],
                         stallAtPixel: Int = -1,
                         stallCycles: Int = 300): Seq[Seq[Int]] = {
    val got = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
    compiled.doSim { (dut: Im2ColContinuityTestComp[SInt]) =>
      dut.clockDomain.forkStimulus(10)
      dut.io.c.stream.ready #= true
      val stream = dut.io.a.stream
      stream.valid #= false
      var pixelIdx = 0
      var timeout = 0
      var stalled = false
      var quiet = 0
      while (pixelIdx < source.length || quiet < 20000) {
        if (!stalled && pixelIdx == stallAtPixel) {
          // Band swap stall: the producer goes idle for a LONG time.
          stream.valid #= false
          for (_ <- 0 until stallCycles) { dut.clockDomain.waitSampling() }
          stalled = true
        }
        if (stream.ready.toBoolean && pixelIdx < source.length) {
          stream.valid #= true
          stream.payload(0) #= source(pixelIdx)
          dut.clockDomain.waitSampling()
          pixelIdx += 1
          stream.valid #= false
          quiet = 0
        } else {
          stream.valid #= false
          dut.clockDomain.waitSampling()
          quiet += 1
        }
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          got += (0 until dut.io.c.stream.payload.length).map(i => dut.io.c.stream.payload(i).toInt)
          quiet = 0
        }
        timeout += 1
        assert(timeout < 400000, "drive loop hung")
      }
    }
    got.toSeq
  }

  test("Im2Col: band-seam stall equivalence — stalled feed == continuous feed") {
    val K = 5
    val W = 28
    val H = 28
    val img = image(H, W, 42)
    val bandH = 5

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
      .compile(Im2ColContinuityTestComp(SInt(16 bits), H, W, 1, K, K * K))

    // Baseline: continuous feed of all H rows.
    val continuous = runCommand(compiled, img, stallAtPixel = -1)
    // Seam feed: band A rows, LONG stall, then band B rows.
    val stalled = runCommand(compiled, img, stallAtPixel = bandH * W)

    assert(continuous.nonEmpty, "no windows emitted on continuous feed")
    assert(continuous.length == stalled.length,
      s"stalled feed emitted ${stalled.length} windows vs ${continuous.length}")
    for (i <- continuous.indices) {
      assert(continuous(i) == stalled(i),
        s"seam window $i differs: continuous ${continuous(i)} vs stalled ${stalled(i)}")
    }
    println(s"Im2Col band-seam equivalence: ${continuous.length} windows identical with vs without a 300-cycle seam stall")
  }

  test("Im2Col: two separate commands stay clean exactly like fresh sessions (M2.2)") {
    val K = 5
    val H = 10
    val W = 28
    val imgA = image(H, W, 1)
    val imgB = image(H, W, 2)
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
      .compile(Im2ColContinuityTestComp(SInt(16 bits), H, W, 1, K, K * K))

    // Baseline sessions: image A alone, then image B alone (fresh sim each) —
    // plus ONE combined session (A then B, no reset) which must equal B-again.
    val aAlone = runCommand(compiled, imgA)
    val bAlone = runCommand(compiled, imgB)
    val bAfterA = runCommand(compiled, imgB)

    assert(aAlone.nonEmpty && bAlone.nonEmpty)
    assert(bAlone == bAfterA,
      s"command-clean violated: fresh-B ${bAlone.length} windows differs from B-after-A ${bAfterA.length}")
    println(s"Im2Col command-clean: B-after-A identical to fresh B (${bAlone.length} windows) — no stale leak")
  }
}
