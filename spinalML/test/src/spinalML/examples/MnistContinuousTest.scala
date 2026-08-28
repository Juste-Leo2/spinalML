package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinal.lib.bus.amba4.axilite.AxiLite4

/**
 * Phase-3 S1 regression: CONTINUOUS run control (streaming execution model).
 *
 * Contract under test:
 *   CSR 0x10 = 0 (STREAM_PER_PASS), CSR 0x1C bit0 = RUN=1, ONE START write.
 *   The accelerator runs inference k = 0.. : on every frame-complete pulse
 *   the hardware re-fires START and ADVANCES the image-base cursor by one
 *   image (video-frame semantics) — the host only pre-writes N contiguous
 *   images and programs 0x08 once. No host↔hardware race exists: the cursor
 *   moves on the same edge as the auto-START, strictly before any DMA reads
 *   it.
 *
 * The gate proves:
 *   - consecutive frames bit-exact vs the JVM replica with ZERO host
 *     involvement after the first START;
 *   - STOP (0x1C = 0) written right after frame 2 lands during the frame
 *     that the engine has ALREADY auto-started (in-flight), then the engine
 *     goes silent forever (40k quiet cycles sampled);
 *   - TILE_CNT (0x18) matches the collected frame count and imgBase (0x08)
 *     reads back exactly base + frameCount * imageBytes.
 *
 * Env: MNIST_CONT_N (default 6 pre-written images), MNIST_CONT_SEED (random
 * image selection), W4A8_ONLY skips the BF16 body.
 */
class MnistContinuousTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val imgBase = 0x10000L
  val weightBase = 0x20000L

  val nImages = sys.env.get("MNIST_CONT_N").map(_.toInt).getOrElse(6)
  val stopAfterNumFrames = 3 // STOP lands while frame #2 is still in flight

  private def selectedImages(n: Int): Seq[Int] =
    sys.env.get("MNIST_CONT_SEED").map(_.toInt) match {
      case Some(seed) =>
        val rng = new scala.util.Random(seed)
        Seq.fill(n)(rng.nextInt(MnistData.images.length))
      case None => Seq.tabulate(n)(k => k % MnistData.images.length)
    }

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  /** Accessor bundle injected per concrete model post-compilation. */
  final case class ContinuityGlue(
    ctrlBus: AxiLite4,
    axiMaster: Axi4,
    outValid: () => Boolean,
    outReady: () => Boolean,
    extractLogit: () => Float)

  private def writeAxiLite(bus: AxiLite4, cd: ClockDomain)(addr: BigInt, data: BigInt): Unit = {
    bus.aw.valid #= true
    bus.aw.payload.addr #= addr
    bus.w.valid #= true
    bus.w.payload.data #= data
    bus.w.payload.strb #= 0xF
    bus.b.ready #= true
    cd.waitSamplingWhere(bus.aw.ready.toBoolean && bus.w.ready.toBoolean)
    bus.aw.valid #= false
    bus.w.valid #= false
    cd.waitSamplingWhere(bus.b.valid.toBoolean)
    bus.b.ready #= false
    cd.waitSampling()
  }

  private def readAxiLite(bus: AxiLite4, cd: ClockDomain)(addr: BigInt): BigInt = {
    bus.ar.valid #= true
    bus.ar.payload.addr #= addr
    bus.r.ready #= true
    cd.waitSamplingWhere(bus.ar.ready.toBoolean)
    bus.ar.valid #= false
    cd.waitSamplingWhere(bus.r.valid.toBoolean)
    val data = bus.r.payload.data.toBigInt
    bus.r.ready #= false
    cd.waitSampling()
    data
  }

  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  private def continuousBody(
    compiled: SimCompiled[_ <: Component],
    glueOf: AnyRef => ContinuityGlue,
    weightWordsOf: () => Seq[BigInt],
    imageWordsOf: Int => Seq[BigInt],
    replica: Seq[String] => Seq[Double],
    tag: String): Unit = {

    val cases = selectedImages(nImages)
    val imgBytes = imageWordsOf(cases.head).length * 8

    compiled.doSim { rawDut =>
      val glue = glueOf(rawDut)
      import glue.outValid, glue.outReady, glue.extractLogit, glue.ctrlBus, glue.axiMaster
      val cd = rawDut.asInstanceOf[Component].clockDomain
      cd.forkStimulus(10)
      val memorySim = AxiMemorySim(axi = axiMaster, clockDomain = cd,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()

      // Weights: one contiguous image pool written BEFORE the first START.
      writeWords(memorySim.memory, weightBase, weightWordsOf())
      for (k <- 0 until nImages) {
        writeWords(memorySim.memory, imgBase + k * imgBytes, imageWordsOf(cases(k)))
      }

      val csr = writeAxiLite(ctrlBus, cd) _
      val rd = readAxiLite(ctrlBus, cd) _
      csr(BigInt(0x08), BigInt(imgBase))
      csr(BigInt(0x0C), BigInt(weightBase))
      csr(BigInt(0x10), 0)

      val frames = scala.collection.mutable.ArrayBuffer[Int]()
      var beats = Seq[Float]()
      var stopIssued = false
      var stoppedAt = -1
      var silence = 0L
      var timeout = 0L
      val maxSamples = sys.env.get("MNIST_TIMEOUT").map(_.toLong).getOrElse(5000000L)

      // ONE start, RUN=1 — then zero host involvement until frame 2.
      csr(BigInt(0x1C), 1)
      csr(BigInt(0x00), 1)
      cd.waitSampling(5)

      while (silence < 40000 && timeout < maxSamples) {
        timeout += 1
        if (outValid() && outReady()) {
          silence = 0
          beats = beats :+ extractLogit()
          if (beats.length == 10) {
            val k = frames.length
            frames += k
            val expected = replica(MnistData.images(cases(k % cases.length)))
            val dev = beats.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
            assert(dev == 0.0,
              s"[$tag frame#$k image#${cases(k % cases.length)}] corrupted under auto-run: hw ${beats.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
            println(f"[$tag/frame#$k] image#${cases(k % cases.length)} predicted ${beats.indexOf(beats.max)}" +
              f" (label ${MnistData.labels(cases(k % cases.length))}) max|hw-sw|=$dev%.3f")
            beats = Nil
            if (!stopIssued && frames.length == stopAfterNumFrames) {
              // In-flight stop: frame k was auto-started by the same done edge
              // that ends frame k−1; RUN=0 lands mid-frame, so exactly ONE
              // extra frame is already committed — then silence.
              csr(BigInt(0x1C), 0)
              stopIssued = true
              stoppedAt = k
            }
          }
        } else {
          silence += 1
        }
        cd.waitSampling()
      }

      assert(stopIssued, s"[$tag] collector never reached frame ${stopAfterNumFrames - 1} — contract failed")
      val expectedFrames = stoppedAt + 2
      assert(frames.length == expectedFrames,
        s"[$tag] expected $expectedFrames frames after in-flight stop, got ${frames.length}")
      val tileCnt = rd(BigInt(0x18))
      assert(tileCnt == frames.length,
        s"[$tag] TILE_CNT=$tileCnt != collected frames=${frames.length}")
      // The CSR readback stays the host-set base; the walker is internal
      // (effective address per frame = base + frameCount × imageBytes).
      val imgBaseBack = rd(BigInt(0x08))
      assert(imgBaseBack == imgBase,
        s"[$tag] imgBase CSR=$imgBaseBack != host base $imgBase (cursor must stay internal)")
      println(s"[$tag] auto-run: ${frames.length} frames bit-exact, TILE_CNT=$tileCnt, imgBase CSR untouched — STOP clean")
    }
  }

  test("Mnist BF16: continuous RUN auto-advances bit-exact frames, STOP is clean") {
    assume(!sys.env.contains("W4A8_ONLY"), "W4A8_ONLY set - skipping BF16 continuous body")
    val bench = new MnistTest
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnist(axiConfig))
    continuousBody(
      compiled,
      d => {
        val dut = d.asInstanceOf[Mnist]
        // Master-stream ready must be DRIVEN by the harness (undriven =
        // random Verilator INIT = seed-dependent deadlock, see Annexe B).
        dut.io.outStream.stream.ready #= true
        ContinuityGlue(dut.io.ctrlBus, dut.io.axiMaster,
          () => dut.io.outStream.stream.valid.toBoolean,
          () => dut.io.outStream.stream.ready.toBoolean,
          () => bench.getFloat(dut.io.outStream.stream.payload(0)))
      },
      () => bench.weightWords(),
      idx => bench.imageWords(MnistData.images(idx)),
      MnistReplica.logits _, tag = "BF16")
  }

  test("Mnistw4a8: continuous RUN auto-advances bit-exact frames, STOP is clean") {
    val bench = new Mnistw4a8Test
    val w4Lanes = W4A8Knob.lanes()
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(W4A8Knob.make(axiConfig))
    continuousBody(
      compiled,
      d => {
        val dut = d.asInstanceOf[Mnistw4a8]
        dut.io.outStream.stream.ready #= true
        ContinuityGlue(dut.io.ctrlBus, dut.io.axiMaster,
          () => dut.io.outStream.stream.valid.toBoolean,
          () => dut.io.outStream.stream.ready.toBoolean,
          () => bench.getFloat(dut.io.outStream.stream.payload(0)))
      },
      () => bench.weightWords(),
      idx => bench.toWords(bench.imageBytes(MnistData.images(idx))),
      (img: Seq[String]) => Mnistw4a8Replica.logitsK(img, w4Lanes), tag = "W4A8")
  }
}
