package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.{BF16, FloatML}
import spinalML.utils.MemLayout
import spinalML.nn.{Accelerator, Linear => LinearSpec}
import spinalML.utils.SimLog

/**
 * Black-box SoC validation of the Mnist accelerator under Verilator.
 *
 * Inputs are pushed through DDR exactly like a real inference (image base +
 * weight base programmed over AXI4-Lite, start bit, output stream collected)
 * and the 10 collected logits are checked against a bit-exact JVM replica of
 * the quantized forward pass ([[MnistReplica]]) — arbitrary inputs therefore
 * become valid test vectors. Curated digits additionally check argmax against
 * the true label.
 *
 * Input selection (environment variables, defaults preserve the historical
 * 5-digit curated run):
 *   MNIST_INDICES="3,7"   subset of the curated digits (labels checked)
 *   MNIST_RANDOM_N=10     random vectors (perturbed digits + pure noise),
 *                         oracle = [[MnistReplica]] only
 *   MNIST_SEED=42         seed for the random generator (default fixed)
 *
 * Each inference runs in its own doSim: the datapath honours the one-shot
 * contract (buffers and FSMs are not re-armed between starts yet), so every
 * case gets a fresh simulation state.
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

  // ------------------------------------------------------------------
  // Dynamic input selection
  // ------------------------------------------------------------------
  case class Tc(image: Seq[String], label: Option[Int], name: String)

  def buildCases(): Seq[Tc] = {
    val curated = MnistData.images.indices.map(i =>
      Tc(MnistData.images(i), Some(MnistData.labels(i)), s"curated#$i"))

    sys.env.get("MNIST_INDICES").map { spec =>
      spec.split(',').map(_.trim.toInt).map(idx => curated(idx)).toSeq
    }.getOrElse {
      sys.env.get("MNIST_RANDOM_N").map(_.toInt).map { n =>
        val rng = new java.util.Random(sys.env.get("MNIST_SEED").map(_.toLong).getOrElse(0x5EEDL))
        Seq.fill(n) {
          if (rng.nextBoolean()) {
            // Perturbed digit: keeps the input near the digit manifold
            val base = MnistData.images(rng.nextInt(MnistData.images.length))
            val img = base.map(row =>
              row.map(c => if (rng.nextInt(100) < 8) (if (c == '1') '0' else '1') else c))
            Tc(img, None, "perturbed")
          } else {
            // Pure noise: pure replica-oracle coverage
            val img = Seq.fill(28)(Seq.fill(28)(
              if (rng.nextInt(100) < 15) '1' else '0').mkString)
            Tc(img, None, "noise")
          }
        }
      }.getOrElse(curated)
    }
  }

  /** One full inference protocol; returns the 10 collected logits. */
  def runInference(dut: Accelerator[FloatML], mem: SparseMemory, image: Seq[String],
                   writeAxiLite: (BigInt, BigInt) => Unit): Seq[Float] = {
    writeWords(mem, imgBase, imageWords(image))

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
    val maxCycles = sys.env.get("MNIST_TIMEOUT").map(_.toInt).getOrElse(5000000)
    while (collected.length < 10 && timeout < maxCycles) {
      if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
        collected += getFloat(dut.io.outStream.stream.payload(0))
      }
      dut.clockDomain.waitSampling()
      timeout += 1
    }

    assert(collected.length == 10,
      s"timeout after $timeout cycles (${collected.length}/10 output beats)")
    collected.toSeq
  }

  test("Mnist SoC black-box: logits match the software replica") {
    val cases = buildCases()
    // The Linear K-chunk width of the model (default weightLanes = inFeatures);
    // MNIST_WLANES overrides it (M2).
    val spec = sys.env.get("MNIST_WLANES") match {
      case Some(s) => Mnist.defaultModelSpec.map {
        case l: LinearSpec => l.copy(weightLanes = s.toInt)
        case o => o
      }
      case None => Mnist.defaultModelSpec
    }
    val wLanes = spec.collectFirst { case l: LinearSpec => l.effLanes }.getOrElse(288)
    val temporal = sys.env.get("MNIST_TEMPORAL").map(_.toInt).getOrElse(0)
    SimLog.info("MNIST")(s"Mnist model wLanes=$wLanes temporal=$temporal (inFeatures=288, cases=${cases.size})")

    // The 288-lane BF16 weight beats are 4608 bits wide, above the default
    // bitVectorWidthMax sanity limit (4096); raise it for this wide model.
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    // NOTE: the accelerator must be constructed INSIDE the (by-name) compile
    // generator — outside it there is no elaboration context for Component.
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Accelerator(dataType = BF16(), inputShape = Seq(28, 28, 1), modelSpec = spec, axiConfig = axiConfig, temporal = temporal))

    var maxDev = 0.0
    for (tc <- cases) {
      compiled.doSim { dut =>
        dut.clockDomain.forkStimulus(10)

        val memorySim = AxiMemorySim(
          axi = dut.io.axiMaster,
          clockDomain = dut.clockDomain,
          config = AxiMemorySimConfig(maxOutstandingReads = 8)
        )
        memorySim.start()

        writeWords(memorySim.memory, weightBase, weightWords())

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

        val logits = runInference(dut, memorySim.memory, tc.image, writeAxiLite)

        val expected = MnistReplica.logitsK(tc.image, wLanes)
        val dev = logits.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        maxDev = math.max(maxDev, dev)
        assert(dev <= sys.env.get("REPLICA_TOL").map(_.toDouble).getOrElse(0.0),
          s"[${tc.name}] logit mismatch vs replica: hw ${logits.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")

        val pred = logits.indexOf(logits.max)
        println(f"[${tc.name}] predicted $pred  logits: ${logits.map(p => f"$p%.3f").mkString("[", " ", "]")}")

        tc.label.foreach { l =>
          assert(pred == l, s"[${tc.name}]: predicted $pred, label $l")
        }
      }
    }

    println(s"Mnist: ${cases.size} inferences checked (max |hw-sw| deviation = $maxDev)")
  }
}
