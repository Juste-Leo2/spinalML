package spinalML.heavy

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.utils.MemLayout
import spinalML.dtypes.FloatML

/**
 * Phase-3 S3/D4 gate: WideConv 64x64 under vertical band tiling.
 *
 * The 64x64 image is genuinely larger than MNIST's 28x28: the benchmark runs
 * the SAME model with tileHeight = 64 (one band, legacy) and tileHeight = 16
 * (4 bands — several band boundaries crossed mid-inference), and both must
 * deliver the exact JVM-replica logits. This validates the banded DMA path
 * on a wide geometry (4-beat-aligned by 64-bit AXI, K=3 windows) where the
 * halo-carried state necessarily spans two + seams.
 */
class WideConvTilingTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  private val imgBase = 0x10000L
  private val weightBase = 0x20000L

  // ---- Bench helpers (same idioms as MnistTest) ----
  private def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
  private def word(elems: Seq[Int]): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0))((acc, e) => acc | (BigInt(e._1 & 0xFFFF) << (16 * e._2)))
  private def packFloats(values: Seq[Float]): Seq[BigInt] =
    values.grouped(4).map(g => word(g.map(bf16Bits).padTo(4, 0))).toSeq

  private def padded(elems: Seq[Float]): Seq[Float] = {
    val capacity = MemLayout.alignToBeat(MemLayout.regionBytes(elems.length, 16), 8) / 2
    elems ++ Seq.fill(capacity - elems.length)(0.0f)
  }

  private def weightWords(): Seq[BigInt] =
    packFloats(padded(WideConvWeights.convW) ++ padded(WideConvWeights.convB) ++
      padded(WideConvWeights.fcW.flatten) ++ padded(WideConvWeights.fcB))

  private def imageWords(img: Seq[String]): Seq[BigInt] =
    packFloats(img.flatMap(_.map(c => if (c == '1') 1.0f else 0.0f)))

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  private def randomImage(seed: Long): Seq[String] = {
    val rng = new scala.util.Random(seed)
    Seq.fill(64)(Seq.fill(64)(if (rng.nextInt(2) == 0) '0' else '1').mkString)
  }

  private def getFloat(p: Data): Float = {
    val f = p.asInstanceOf[FloatML]
    val bits = ((if (f.sign.toBoolean) 1 else 0) << 15) | ((f.exponent.toInt & 0xFF) << 7) | (f.mantissa.toInt & 0x7F)
    java.lang.Float.intBitsToFloat(bits << 16)
  }

  def runInference(dut: WideConv, mem: SparseMemory, img: Seq[String],
                   writeAxiLite: (BigInt, BigInt) => Unit): Seq[Float] = {
    writeWords(mem, imgBase, imageWords(img))
    dut.io.ctrlBus.aw.valid #= false
    dut.io.ctrlBus.w.valid #= false
    dut.io.ctrlBus.ar.valid #= false
    dut.io.ctrlBus.r.ready #= false
    dut.io.ctrlBus.b.ready #= false
    dut.io.outStream.stream.ready #= true
    dut.clockDomain.waitSampling(5)
    writeAxiLite(BigInt(0x08), BigInt(imgBase))
    writeAxiLite(BigInt(0x0C), BigInt(weightBase))
    writeAxiLite(BigInt(0x00), 1)

    val collected = scala.collection.mutable.ArrayBuffer[Float]()
    var timeout = 0
    // WideConv passes are ~100k cycles; an env override exists for CI budget.
    val maxCycles = sys.env.get("MNIST_TIMEOUT").map(_.toInt).getOrElse(800000)
    while (collected.length < 10 && timeout < maxCycles) {
      if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
        collected += getFloat(dut.io.outStream.stream.payload(0))
      }
      dut.clockDomain.waitSampling()
      timeout += 1
    }
    assert(collected.length == 10, s"WideConv timeout after $timeout cycles")
    collected.toSeq
  }

  test("WideConv BF16: 64x64 tiled (tileHeight 64 vs 16) matches the replica") {
    val images = Seq(randomImage(7), randomImage(23), randomImage(99))
    for (tileH <- Seq(64, 16)) {
      val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
        .compile(WideConv(axiConfig, tileHeight = tileH))
      compiled.doSim { dut: WideConv =>
        dut.clockDomain.forkStimulus(10)
        val memorySim = AxiMemorySim(axi = dut.io.axiMaster, clockDomain = dut.clockDomain,
          config = AxiMemorySimConfig(maxOutstandingReads = 8))
        memorySim.start()
        writeWords(memorySim.memory, weightBase, weightWords())

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

        for ((img, k) <- images.zipWithIndex) {
          val got = runInference(dut, memorySim.memory, img, writeAxiLite)
          val expected = WideConvReplica.logits(img)
          val dev = got.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
          assert(dev == 0.0,
            s"[WideConv tileH=$tileH image#$k] corrupted: hw ${got.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
          println(f"[WideConv tileH=$tileH%-3d image#$k predicted ${got.indexOf(got.max)} max|hw-sw|=$dev%.3f")
        }
      }
      println(s"WideConv tileHeight=$tileH: ${images.length} x64x64 images bit-exact")
    }
  }
}
