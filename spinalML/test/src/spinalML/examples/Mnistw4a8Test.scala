package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.FloatML
import spinalML.utils.MemLayout

/**
 * Black-box SoC validation of the W4A8 Mnist accelerator under Verilator.
 *
 * Same protocol as [[MnistTest]] (no golden model: the trained network IS the
 * reference), but the DDR image is mixed-precision:
 *   ConvW  : raw int4 codes packed two-per-nibble-pair, 16 elements per beat
 *            (the DMA slices element i at bit 4*i of the AXI word);
 *   ConvB  : int-domain biases (b_q = round(b / convScale)) as I16 little-endian;
 *   FcW/FcB: E4M3 bytes (weights are grid-exact, bias rounded to nearest);
 *   Image  : raw 0/1 integer byte codes (the conv runs on integer
 *            activations, NOT float-encoded pixels).
 * Each weight section is padded to a whole 64-bit word, mirroring the
 * builder's beat-aligned region layout.
 */
class Mnistw4a8Test extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val imgBase = 0x10000
  val weightBase = 0x20000
  val imgStride = 28 * 28 // one image: 784 E4M3 bytes, already word-aligned

  /** Encodes a float to the nearest E4M3 byte (normals and subnormals). */
  def fp8Bits(f: Float): Int = {
    if (f == 0f) return 0
    val sign = if (f < 0) 0x80 else 0
    val a = math.abs(f.toDouble)
    if (a >= math.pow(2, -6)) { // normal: exp 4 bits (bias 7), mantissa 3 bits
      var e = math.floor(math.log(a) / math.log(2)).toInt
      var mF = a / math.pow(2, e)
      if (mF >= 2) { mF /= 2; e += 1 }
      var m = math.floor((mF - 1) * 8 + 0.5).toInt
      if (m == 8) { m = 0; e += 1 }
      val eb = e + 7
      assert(eb >= 1 && eb <= 15, s"$f outside E4M3 normal range")
      sign | (eb << 3) | m
    } else {                    // subnormal: step 2^-9
      val m = math.floor(a * 512 + 0.5).toInt
      assert(m <= 7, s"$f overflows subnormal range")
      sign | m
    }
  }

  /** Decodes any FloatML payload from its actual field widths. */
  def getFloat(p: Data): Float = {
    val f = p.asInstanceOf[FloatML]
    val eW = f.exponent.getWidth
    val mW = f.mantissa.getWidth
    val sign = if (f.sign.toBoolean) -1 else 1
    val rawE = f.exponent.toInt
    val rawM = f.mantissa.toInt
    val bias = (1 << (eW - 1)) - 1
    val mag =
      if (rawE == 0) rawM * math.pow(2, 1 - bias - mW)
      else (1 + rawM.toDouble / (1 << mW)) * math.pow(2, rawE - bias)
    (sign * mag).toFloat
  }

  /** Little-endian packers: one element per slot of the given width. */
  def wordOf(bytes: Seq[Int], widthBits: Int): BigInt =
    bytes.zipWithIndex.foldLeft(BigInt(0))((acc, e) =>
      acc | (BigInt(e._1 & ((1 << widthBits) - 1)) << (widthBits * e._2)))

  def writeWords(mem: SparseMemory, base: Int, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  /** Pads a byte stream up to whole 64-bit AXI words (builder beat alignment). */
  def toWords(bytes: Seq[Int]): Seq[BigInt] =
    bytes.grouped(8).map(g => wordOf(g.padTo(8, 0), 8)).toSeq

  val beatBytes = 8 // axiConfig.dataWidth / 8; benches assume a 64-bit bus

  /** One weight region exactly as the builder lays it out: size from
    * MemLayout.regionBytes (whole-region ceil, sub-byte safe) rounded up to
    * the AXI beat — bench and RTL share a single layout convention. */
  def region(bytes: Seq[Int], elements: Int, elemBits: Int): Seq[BigInt] = {
    val size = MemLayout.alignToBeat(MemLayout.regionBytes(elements, elemBits), beatBytes)
    require(bytes.length <= size, s"weight region overflow: ${bytes.length}B > $size B")
    toWords(bytes.padTo(size, 0))
  }

  def nibbleBytes(qs: Seq[Int]): Seq[Int] =
    qs.grouped(2).map(g => (g.head & 0xF) | ((g.apply(1) & 0xF) << 4)).toSeq

  /** INT-domain pixels: raw 0/1 byte codes (the conv runs on integer activations). */
  def imageBytes(rows: Seq[String]): Seq[Int] =
    rows.flatMap(_.map(c => if (c == '1') 1 else 0))

  /**
   * Weight region in builder order (Sequential stacks per layer: weights then
   * bias, layers in declaration order, each region beat-aligned):
   *   ConvW [50 x I4] | pad | ConvB [2 x I16] | pad | FcW [2880 x E4M3] | pad | FcB [10 x E4M3] | pad
   */
  def weightWords(): Seq[BigInt] = {
    val wq = Mnistw4a8Weights.convWq.flatten
    val fw = Mnistw4a8Weights.fcW.flatten
    val convW = region(nibbleBytes(wq), wq.length, 4)
    val convB = region(
      Mnistw4a8Weights.convBq.flatMap(v => Seq(v & 0xFF, (v >> 8) & 0xFF)),
      Mnistw4a8Weights.convBq.length, 16)
    val fcW = region(fw.map(v => fp8Bits(v)), fw.length, 8)
    val fcB = region(Mnistw4a8Weights.fcB.map(v => fp8Bits(v)), Mnistw4a8Weights.fcB.length, 8)
    convW ++ convB ++ fcW ++ fcB
  }

  test("Mnistw4a8 SoC black-box: 5/5 digits classified end to end") {
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnistw4a8(axiConfig))

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
          toWords(imageBytes(MnistData.images(idx))))

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

        val logits = collected.toSeq
        val pred = logits.indexOf(logits.max)
        val shown = logits.map(p => f"$p%.3f").mkString("[", " ", "]")
        println(f"Image $idx -> predicted $pred, label ${MnistData.labels(idx)}  logits: $shown")

        assert(pred == MnistData.labels(idx),
          s"Image $idx: predicted $pred, label ${MnistData.labels(idx)}")
      }
      correct += 1
    }

    println(s"Mnistw4a8: $correct/${MnistData.images.length} digits correctly classified")
    assert(correct == MnistData.images.length, s"$correct/${MnistData.images.length} only")
  }
}
