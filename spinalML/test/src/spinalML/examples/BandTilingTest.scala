package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.nn.Accelerator

/**
 * Phase-3 S3 gate: vertical BAND tiling of the image path (streaming model).
 *
 * The SAME model is compiled with tileHeight = 28 (band = whole image,
 * legacy one-shot) and tileHeight = 14 / 10 (multiple bands per inference:
 * back-to-back 2D patch commands, on-chip image buffer sized to ONE band,
 * row stream continuous across band seams). Every variant runs several
 * images through runInference and must deliver the EXACT JVM-replica logits:
 * bit-exactness of the banded path is the end-to-end proof that the seam
 * state carrying (the im2col halo contract proven op-level in S2) composes
 * with real DMA/arbiter/buffer timing.
 */
class BandTilingTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  private val bandCases = Seq(28, 14, 10)
  private val imgBase = 0x10000L

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  private def bandedCase[M <: Accelerator[_]](
    makeModel: Int => M,
    weightWordsOf: () => Seq[BigInt],
    imageWordsOf: (SparseMemory, Int) => Unit,
    logitsOf: (M, SparseMemory, Seq[String], (BigInt, BigInt) => Unit) => Seq[Float],
    replicaOf: Seq[String] => Seq[Double],
    tag: String): Unit = {

    val cases = Seq(0, 1, 2)
    for (tileH <- bandCases) {
      val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeModel(tileH))
      compiled.doSim { dut =>
        dut.clockDomain.forkStimulus(10)
        val memorySim = AxiMemorySim(
          axi = dut.io.axiMaster,
          clockDomain = dut.clockDomain,
          config = AxiMemorySimConfig(maxOutstandingReads = 8))
        memorySim.start()
        writeWords(memorySim.memory, 0x20000L, weightWordsOf())

        def writeAxiLite(addr: BigInt, data: BigInt): Unit = {
          dut.io.ctrlBus.aw.valid #= true
          dut.io.ctrlBus.aw.payload.addr #= addr
          dut.io.ctrlBus.w.valid #= true
          dut.io.ctrlBus.w.payload.data #= data
          dut.io.ctrlBus.w.payload.strb #= 0xF
          dut.io.ctrlBus.b.ready #= true
          dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
          dut.io.ctrlBus.aw.valid #= false
          dut.io.ctrlBus.w.valid #= false
          dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
          dut.io.ctrlBus.b.ready #= false
          dut.clockDomain.waitSampling()
        }

        for (k <- cases) {
          val img: Seq[String] = MnistData.images(k)
          writeWords(memorySim.memory, imgBase, Seq(BigInt(0))) // no-op, runInference writes the image
          val got = logitsOf(dut, memorySim.memory, img, writeAxiLite)
          val expected = replicaOf(img)
          val dev = got.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
          assert(dev == 0.0,
            s"[$tag tileH=$tileH image#$k] corrupted: hw ${got.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
          println(f"[$tag tileH=$tileH%-3d image#$k predicted ${got.indexOf(got.max)}" +
            f" (label ${MnistData.labels(k)}) max|hw-sw|=$dev%.3f")
        }
      }
      println(s"[$tag] tileHeight=$tileH: ${cases.length} images bit-exact")
    }
  }

  test("Mnist BF16: banded tileHeight variants match the replica (differential vs full)") {
    assume(!sys.env.contains("W4A8_ONLY"), "W4A8_ONLY set - skipping BF16 band tiling")
    val bench = new MnistTest
    bandedCase[Mnist](
      tileH => Mnist(axiConfig, tileHeight = tileH),
      () => bench.weightWords(),
      (mem, idx) => writeWords(mem, imgBase, bench.imageWords(MnistData.images(idx))),
      (dut, mem, img, csr) => bench.runInference(dut, mem, img, csr),
      MnistReplica.logits _, tag = "BF16")
  }

  test("Mnistw4a8: banded tileHeight variants match the replica (differential vs full)") {
    val bench = new Mnistw4a8Test
    bandedCase[Mnistw4a8](
      tileH => Mnistw4a8(axiConfig, tileHeight = tileH),
      () => bench.weightWords(),
      (mem, idx) => writeWords(mem, imgBase, bench.toWords(bench.imageBytes(MnistData.images(idx)))),
      (dut, mem, img, csr) => bench.runInference(dut, mem, img, csr),
      Mnistw4a8Replica.logitsK(_, 4), tag = "W4A8")
  }
}
