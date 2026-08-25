package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.FloatML
import spinalML.utils.MemLayout

/**
 * Black-box SoC validation of the Mnist accelerator under Verilator.
 *
 * The five binarized digits are pushed through DDR exactly like a real
 * inference (image base + weight base programmed over AXI4-Lite, start bit,
 * output stream collected), and only the final argmax of the 10 logits is
 * checked against the true labels. No golden model involved: the trained
 * network IS the reference.
 *
 * Each image runs in its own doSim: the datapath honours the one-shot
 * inference contract (buffers and FSMs are not re-armed between starts),
 * so every digit gets a fresh simulation state.
 */
class MnistTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val imgBase = 0x10000
  val weightBase = 0x20000
  val imgStride = 28 * 28 * 2 // one image: 784 BF16 elements, word-aligned

  def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF

  /** Packs 16-bit elements into 64-bit AXI words, little-endian (lane 0 = LSB). */
  def word(elems: Seq[Int]): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0))((acc, e) => acc | (BigInt(e._1 & 0xFFFF) << (16 * e._2)))

  def packFloats(values: Seq[Float]): Seq[BigInt] =
    values.grouped(4).map(g => word(g.map(bf16Bits).padTo(4, 0))).toSeq

  def writeWords(mem: SparseMemory, base: Int, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  /** Pads a weight section up to its builder footprint (MemLayout.regionBytes
    * rounded up to the 64-bit AXI beat), mirroring Sequential. */
  def padded(elems: Seq[Float]): Seq[Float] = {
    val capacity = MemLayout.alignToBeat(MemLayout.regionBytes(elems.length, 16), 8) / 2
    elems ++ Seq.fill(capacity - elems.length)(0.0f)
  }

  def imageWords(rows: Seq[String]): Seq[BigInt] =
    packFloats(rows.flatMap(_.map(c => if (c == '1') 1.0f else 0.0f)))

  /**
   * Weight region in builder order (Sequential stacks per layer: weights then
   * bias, layers in declaration order, each region beat-aligned):
   *   ConvW [2 x 25] | pad | ConvB [2] | pad | FcW [10 x 288] | pad | FcB [10]
   */
  def weightWords(): Seq[BigInt] =
    packFloats(padded(MnistWeights.convW.flatten) ++ padded(MnistWeights.convB) ++
      padded(MnistWeights.fcW.flatten) ++ padded(MnistWeights.fcB))

  def getFloat(p: Data): Float = {
    val f = p.asInstanceOf[FloatML]
    val bits = ((if (f.sign.toBoolean) 1 else 0) << 15) | ((f.exponent.toInt & 0xFF) << 7) | (f.mantissa.toInt & 0x7F)
    java.lang.Float.intBitsToFloat(bits << 16)
  }

  test("Mnist SoC black-box: 5/5 digits classified end to end") {
    // The 288-lane BF16 weight beats are 4608 bits wide, above the default
    // bitVectorWidthMax sanity limit (4096); raise it for this wide model.
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnist(axiConfig))

    var correct = 0
    for (idx <- MnistData.images.indices) {
      compiled.doSim { dut =>
        dut.clockDomain.forkStimulus(10)

        val memorySim = AxiMemorySim(
          axi = dut.io.axiMaster,
          clockDomain = dut.clockDomain,
          config = AxiMemorySimConfig(maxOutstandingReads = 8)
        )
        memorySim.start()

        writeWords(memorySim.memory, weightBase, weightWords())
        writeWords(memorySim.memory, imgBase + idx * imgStride,
          imageWords(MnistData.images(idx)))

        def writeAxiLite(addr: BigInt, data: BigInt) = {
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

        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.io.ctrlBus.ar.valid #= false
        dut.io.ctrlBus.r.ready #= false
        dut.io.ctrlBus.b.ready #= false
        dut.io.outStream.stream.ready #= true

        dut.clockDomain.waitSampling(5)

        writeAxiLite(BigInt(0x08), BigInt(imgBase + idx * imgStride))
        writeAxiLite(BigInt(0x0C), BigInt(weightBase))
        writeAxiLite(BigInt(0x00), 1)

        val collected = scala.collection.mutable.ArrayBuffer[Float]()
        var timeout = 0
        while (collected.length < 10 && timeout < 5000000) {
          if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
            collected += getFloat(dut.io.outStream.stream.payload(0))
          }
          dut.clockDomain.waitSampling()
          timeout += 1
        }

        assert(collected.length == 10,
          s"Image $idx: timeout after $timeout cycles (${collected.length}/10 output beats)")

        val probs = collected.toSeq
        val pred = probs.indexOf(probs.max)
        val shown = probs.map(p => f"$p%.3f").mkString("[", " ", "]")
        println(f"Image $idx -> predicted $pred, label ${MnistData.labels(idx)}  logits: $shown")

        assert(pred == MnistData.labels(idx),
          s"Image $idx: predicted $pred, label ${MnistData.labels(idx)}")
      }
      correct += 1
    }

    println(s"Mnist: $correct/${MnistData.images.length} digits correctly classified")
    assert(correct == MnistData.images.length, s"$correct/${MnistData.images.length} only")
  }
}
