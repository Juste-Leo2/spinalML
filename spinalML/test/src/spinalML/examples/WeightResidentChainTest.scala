package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.{Axi4, Axi4Config, Axi4ReadOnly}
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}

/**
 * Weight-residency regression (Phase 2a — roadmap "Weight Manager" first half).
 *
 * Protocol inside ONE simulation session, no reset ever:
 *   Pass 0   : mode STREAM_PER_PASS (CSR 0x10 = 0) — legacy baseline, weight
 *              AND image AXI bursts expected, logits vs JVM replica.
 *   Mode set : CSR 0x10 bit0 = 1 (WEIGHT_RESIDENT).
 *   Pass 1   : the rising edge of resident mode self-triggers exactly one
 *              guaranteed weight refetch (the compute pointer otherwise sits
 *              on a flipped-empty bank left by the baseline pass), so
 *              weight ARs > 0 are EXPECTED here.
 *   Pass 2.. : steady resident state — weight-region ARs must be STRICTLY
 *              ZERO while activations keep flowing; logits stay bit-exact
 *              against the replica.
 *   RELOAD   : one write to CSR 0x14 arms every region for exactly one
 *              refetch at the next START (anti-vacuity of the reload path),
 *              after which zero-weight-traffic residency resumes.
 *
 * Environment knobs: MNIST_RESIDENT_N (steady STARTs, default 3),
 * MNIST_RESIDENT_SEED (reproducible digit itinerary), W4A8_ONLY skips BF16.
 */
class WeightResidentChainTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val imgBase = 0x10000L
  val weightBase = 0x20000L

  val nSteady = sys.env.get("MNIST_RESIDENT_N").map(_.toInt).getOrElse(3)

  /** Pass itinerary: seeded pseudo-random digits (default: cycle 0,1,2,...). */
  private def passCases(n: Int): Seq[Int] =
    sys.env.get("MNIST_RESIDENT_SEED").map(_.toInt) match {
      case Some(seed) =>
        val rng = new scala.util.Random(seed)
        Seq.fill(n)(rng.nextInt(MnistData.images.length))
      case None => Seq.tabulate(math.max(n, 1))(_ % MnistData.images.length)
    }

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  private def writeAxiLite(dutCd: ClockDomain, bus: AxiLite4)(addr: BigInt, data: BigInt): Unit = {
    bus.aw.valid #= true
    bus.aw.payload.addr #= addr
    bus.w.valid #= true
    bus.w.payload.data #= data
    bus.w.payload.strb #= 0xF
    bus.b.ready #= true
    dutCd.waitSamplingWhere(bus.aw.ready.toBoolean && bus.w.ready.toBoolean)
    bus.aw.valid #= false
    bus.w.valid #= false
    dutCd.waitSamplingWhere(bus.b.valid.toBoolean)
    bus.b.ready #= false
    dutCd.waitSampling()
  }

  /** Passive observer classifying each AXI read address into (image, weight). */
  private class ArMeter(dutCd: ClockDomain, axi: Axi4) {
    private[WeightResidentChainTest] var img = 0L
    private[WeightResidentChainTest] var wt = 0L
    dutCd.onSamplings {
      if (axi.ar.valid.toBoolean && axi.ar.ready.toBoolean) {
        val a = axi.ar.addr.toLong
        if (a >= weightBase) wt += 1
        else if (a >= imgBase) img += 1
      }
    }
    def snap(): (Long, Long) = (img, wt)
  }

  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  // =====================================================================
  // BF16 body
  // =====================================================================
  test("Mnist BF16: residency keeps logits bit-exact with zero weight DDR traffic") {
    assume(!sys.env.contains("W4A8_ONLY"), "W4A8_ONLY set - skipping BF16 residency body")
    val bench = new MnistTest
    val cases = passCases(nSteady + 3)

    SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnist(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      val memorySim = AxiMemorySim(axi = dut.io.axiMaster, clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      writeWords(memorySim.memory, weightBase, bench.weightWords())

      val meter = new ArMeter(dut.clockDomain, dut.io.axiMaster)
      val csr = writeAxiLite(dut.clockDomain, dut.io.ctrlBus) _
      def infer(idx: Int): Seq[Float] =
        bench.runInference(dut, memorySim.memory, MnistData.images(idx), csr)

      def check(step: String, idx: Int, logits: Seq[Float]): Unit = {
        val expected = MnistReplica.logits(MnistData.images(idx))
        val dev = logits.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        assert(dev == 0.0,
          s"[BF16/$step image#$idx] corrupted under residency: hw ${logits.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
        println(f"[BF16/$step%-11s image#$idx predicted ${logits.indexOf(logits.max)} (label ${MnistData.labels(idx)})" +
          f"  max|hw-sw|=$dev%.3f")
      }

      csr(BigInt(0x08), BigInt(imgBase))
      csr(BigInt(0x0C), BigInt(weightBase))
      csr(BigInt(0x10), 0) // explicit STREAM_PER_PASS

      // Pass 0: legacy baseline, every region must hit DDR
      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(0))
        check("baseline", cases(0), l)
        val (aI, aW) = meter.snap()
        val ci = aI - bI; val cw = aW - bW
        assert(ci > 0 && cw > 0, s"[baseline] expected image AND weight bursts (img=$ci w=$cw)")
        println(s"[BF16/baseline ] image ARs=$ci weight ARs=$cw")
      }

      // Enable WEIGHT_RESIDENT
      csr(BigInt(0x10), 1)

      // Pass 1: rising edge guarantees one resident-entry refetch
      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(1))
        check("rise", cases(1), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[rise] image must still stream")
        assert(aW - bW > 0, s"[rise] expected the guaranteed resident-entry refetch")
        println(s"[BF16/rise     ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      // Steady state: STRICTLY ZERO weight-region bursts
      for (k <- 0 until nSteady) {
        val (bI, bW) = meter.snap()
        val l = infer(cases(2 + k))
        check("resident", cases(2 + k), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[resident] image bursts vanished?")
        assert(aW - bW == 0, s"[resident] LEAK: ${aW - bW} weight-region ARs at k=$k")
        println(s"[BF16/resident ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      // RELOAD anti-vacuity: next pass MUST refetch
      csr(BigInt(0x14), BigInt(1)) // any write pulses the shot
      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(nSteady + 2))
        check("reload", cases(nSteady + 2), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[reload] image must still stream")
        assert(aW - bW > 0, "[reload] reload pulse ignored — path dead or not wired")
        println(s"[BF16/reload   ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      // Residency resumes cleanly afterwards
      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(1))
        check("post-reload", cases(1), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[post-reload] image bursts vanished?")
        assert(aW - bW == 0, s"[post-reload] LEAK after reload: ${aW - bW} weight ARs")
        println(s"[BF16/post-rl  ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      println(s"BF16 residency chain OK (${nSteady + 4} passes bit-exact, steady-state weight DDR == 0)")
    }
  }

  // =====================================================================
  // W4A8 body
  // =====================================================================
  test("Mnistw4a8: residency keeps logits bit-exact with zero weight DDR traffic") {
    val bench = new Mnistw4a8Test
    val w4Lanes = W4A8Knob.lanes()
    val cases = passCases(nSteady + 3)

    SimConfig.withVerilator.withConfig(spinalConfig).compile(W4A8Knob.make(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      val memorySim = AxiMemorySim(axi = dut.io.axiMaster, clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      writeWords(memorySim.memory, weightBase, bench.weightWords())

      val meter = new ArMeter(dut.clockDomain, dut.io.axiMaster)
      val csr = writeAxiLite(dut.clockDomain, dut.io.ctrlBus) _
      def infer(idx: Int): Seq[Float] =
        bench.runInference(dut, memorySim.memory, MnistData.images(idx), csr)

      def check(step: String, idx: Int, logits: Seq[Float]): Unit = {
        val expected = Mnistw4a8Replica.logitsK(MnistData.images(idx), w4Lanes)
        val dev = logits.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        assert(dev == 0.0,
          s"[W4A8/$step image#$idx] corrupted under residency: hw ${logits.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
        println(f"[W4A8/$step%-11s image#$idx predicted ${logits.indexOf(logits.max)} (label ${MnistData.labels(idx)})" +
          f"  max|hw-sw|=$dev%.3f")
      }

      csr(BigInt(0x08), BigInt(imgBase))
      csr(BigInt(0x0C), BigInt(weightBase))
      csr(BigInt(0x10), 0)

      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(0))
        check("baseline", cases(0), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0 && aW - bW > 0, s"[baseline] expected image AND weight bursts")
        println(s"[W4A8/baseline ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      csr(BigInt(0x10), 1)

      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(1))
        check("rise", cases(1), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[rise] image must still stream")
        assert(aW - bW > 0, "[rise] expected the guaranteed resident-entry refetch")
        println(s"[W4A8/rise     ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      for (k <- 0 until nSteady) {
        val (bI, bW) = meter.snap()
        val l = infer(cases(2 + k))
        check("resident", cases(2 + k), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[resident] image bursts vanished?")
        assert(aW - bW == 0, s"[resident] LEAK: ${aW - bW} weight-region ARs at k=$k")
        println(s"[W4A8/resident ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      csr(BigInt(0x14), BigInt(1))
      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(nSteady + 2))
        check("reload", cases(nSteady + 2), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[reload] image must still stream")
        assert(aW - bW > 0, "[reload] reload pulse ignored — path dead or not wired")
        println(s"[W4A8/reload   ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      locally {
        val (bI, bW) = meter.snap()
        val l = infer(cases(1))
        check("post-reload", cases(1), l)
        val (aI, aW) = meter.snap()
        assert(aI - bI > 0, "[post-reload] image bursts vanished?")
        assert(aW - bW == 0, s"[post-reload] LEAK after reload: ${aW - bW} weight ARs")
        println(s"[W4A8/post-rl  ] image ARs=${aI - bI} weight ARs=${aW - bW}")
      }

      println(s"W4A8 residency chain OK (${nSteady + 4} passes bit-exact, steady-state weight DDR == 0)")
    }
  }
}
