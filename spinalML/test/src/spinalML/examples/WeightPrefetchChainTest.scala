package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.{Axi4, Axi4Config}
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}

/**
 * Phase-2b weight PREFETCH regression (Weight Manager, second half).
 *
 * Mechanism under test: with RESIDENT+PREFETCH_EN (CSR 0x10 = 0b11), a RELOAD
 * arms regions that fire EAGERLY against reader-ready x loader-empty — filling
 * the IDLE banks while nothing consumes — and stage a governed swap landing at
 * the NEXT end-of-pass edge, never mid-stream.
 *
 * The overlap is proven TRAFFIC-WISE (robust on CI, unlike wall-clock ratios):
 *   - SERIALIZED reload (RESIDENT only): the whole weight region is fetched
 *     BETWEEN the START assertion and the first output beat — weight-region
 *     ARs inside that window == the fixed per-model reference count.
 *   - EAGER reload: those ARs completed during the idle window before START,
 *     so the very same window shows ZERO weight ARs.
 *
 * Relocating weightsBase to an alternate aligned copy between passes makes
 * each eager generation a GENUINE new fetch+swap through fresh addresses;
 * contents stay byte-identical so the JVM replicas remain exact oracles.
 *
 * Env: MNIST_PREFETCH_PAIRS (default 2), MNIST_TIMEOUT, W4A8_ONLY skips BF16.
 */
class WeightPrefetchChainTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val imgBase = 0x10000L
  val baseA = 0x20000L
  val baseB = 0x40000L

  val pairs = sys.env.get("MNIST_PREFETCH_PAIRS").map(_.toInt).getOrElse(2)

  private def itinerary(n: Int): Seq[Int] =
    sys.env.get("MNIST_PREFETCH_SEED").map(_.toInt) match {
      case Some(seed) =>
        val rng = new scala.util.Random(seed)
        Seq.fill(n)(rng.nextInt(MnistData.images.length))
      case None => Seq.tabulate(n)(_ % MnistData.images.length)
    }

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  private def writeAxiLite(cd: ClockDomain, bus: AxiLite4)(addr: BigInt, data: BigInt): Unit = {
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

  private class ArMeter(cd: ClockDomain, axi: Axi4) {
    private[WeightPrefetchChainTest] var img = 0L
    private[WeightPrefetchChainTest] var wt = 0L
    cd.onSamplings {
      if (axi.ar.valid.toBoolean && axi.ar.ready.toBoolean) {
        val a = axi.ar.addr.toLong
        if (a >= baseA) wt += 1
        else if (a >= imgBase) img += 1
      }
    }
    def snap(): (Long, Long) = (img, wt)
  }

  // =====================================================================
  // Generic session over injected accessors (keeps Spinal types out of
  // generics — one closure per concrete model).
  // =====================================================================
  private def prefetchBody(
    compiled: SimCompiled[_ <: Component],
    weightWordsOf: () => Seq[BigInt],
    writeImageOf: (SparseMemory, Seq[String]) => Unit,
    glueOf: AnyRef => PrefetchGlue,
    replica: Seq[String] => Seq[Double],
    tag: String): Unit = {

    compiled.doSim { rawDut =>
      val dut = glueOf(rawDut)
      import dut._
      dutCd.forkStimulus(10)
      val memorySim = AxiMemorySim(axi = axiMaster, clockDomain = dutCd,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      writeWords(memorySim.memory, baseA, weightWordsOf())
      writeWords(memorySim.memory, baseB, weightWordsOf())

      val csr = writeAxiLite(dutCd, ctrlBus) _
      val meter = new ArMeter(dutCd, axiMaster)
      val cases = itinerary(pairs * 2 + 3)

      def runMetered(idx: Int): (Seq[Float], Long, Long) = {
        writeImageOf(memorySim.memory, MnistData.images(idx))
        csr(BigInt(0x00), BigInt(1))
        val (_, bw0) = meter.snap()
        val collected = scala.collection.mutable.ArrayBuffer[Float]()
        var wAtFirstBeat = -1L
        var timeout = 0
        val maxCycles = sys.env.get("MNIST_TIMEOUT").map(_.toInt).getOrElse(5000000)
        while (collected.length < 10 && timeout < maxCycles) {
          if (outValid() && outReady()) {
            collected += extractLogit()
            if (collected.length == 1) wAtFirstBeat = meter.snap()._2 - bw0
          }
          dutCd.waitSampling()
          timeout += 1
        }
        assert(collected.length == 10, s"[$tag image#$idx] timeout after $timeout cycles")
        val (bi1, _) = meter.snap()
        (collected.toSeq, wAtFirstBeat, bi1 - bw0)
      }

      def checkExact(step: String, idx: Int, got: Seq[Float]): Unit = {
        val expected = replica(MnistData.images(idx))
        val dev = got.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        assert(dev == 0.0,
          s"[$tag/$step image#$idx] corrupted under prefetch: hw ${got.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
        println(f"[${tag}/$step%-9s image#$idx predicted ${got.indexOf(got.max)} (label ${MnistData.labels(idx)})" +
          f"  max|hw-sw|=$dev%.3f")
      }

      dutCd.waitSampling(20)

      // ---- Pass 0: STREAM_PER_PASS baseline (serializes everything) -------
      csr(BigInt(0x08), BigInt(imgBase))
      csr(BigInt(0x0C), BigInt(baseA))
      csr(BigInt(0x10), 0)
      locally {
        val (l, _, imgAr) = runMetered(cases(0))
        checkExact("baseline", cases(0), l)
        val cw0 = meter.snap()._2
        assert(imgAr > 0 && cw0 > 0, "[baseline] expected real bursts")
        println(s"[$tag/baseline ] cumulative weight ARs = $cw0")
      }

      // ---- Alternating serial/eager pairs ----------------------------------
      for (p <- 0 until pairs) {
        val idxSlow = cases(1 + p * 2)
        val idxFast = cases(2 + p * 2)

        // SERIALIZED: resident-only; RELOAD fires at the START boundary
        csr(BigInt(0x10), 1)
        csr(BigInt(0x14), BigInt(1))
        locally {
          val (l, wWin, imgAr) = runMetered(idxSlow)
          checkExact(s"serial#$p", idxSlow, l)
          assert(wWin > 0, s"[serial#$p] expected fetches inside the window (got $wWin)")
          assert(imgAr > 0, s"[serial#$p] image bursts vanished?")
          println(s"[$tag/serial#$p ] weight ARs in START→first-beat window = $wWin")
        }

        // EAGER: resident+prefetch; also swap weightsBase for a genuine refetch
        csr(BigInt(0x10), 3)
        csr(BigInt(0x0C), BigInt(if (p % 2 == 0) baseB else baseA))
        csr(BigInt(0x14), BigInt(1))
        cdWaitIdle() // model-supplied settle: let eager fills land & stage
        locally {
          val (l, wWin, imgAr) = runMetered(idxFast)
          checkExact(s"eager#$p", idxFast, l)
          assert(imgAr > 0, s"[eager#$p] image bursts vanished?")
          assert(wWin == 0,
            s"[eager#$p] LEAK: $wWin weight ARs still inside the window — prefetch did not cover the fetch")
          println(s"[$tag/eager#$p  ] weight ARs in START→first-beat window = 0 (overlapped)")
        }
      }

      println(s"$tag prefetch chain OK (${pairs * 2 + 1} passes bit-exact; eager windows carry ZERO weight ARs)")
    }
  }

  /** Accessor bundle extracted post-compilation per concrete model. */
  final case class PrefetchGlue(
    dutCd: ClockDomain,
    ctrlBus: AxiLite4,
    axiMaster: Axi4,
    outValid: () => Boolean,
    outReady: () => Boolean,
    extractLogit: () => Float,
    cdWaitIdle: () => Unit)

  import scala.language.reflectiveCalls

  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  test("Mnist BF16: eager weight prefetch overlaps DDR with compute") {
    assume(!sys.env.contains("W4A8_ONLY"), "W4A8_ONLY set - skipping BF16 prefetch body")
    val bench = new MnistTest
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnist(axiConfig))
    prefetchBody(
      compiled,
      () => bench.weightWords(),
      (mem, img) => writeWords(mem, imgBase, bench.imageWords(img)),
      d => {
        val dut = d.asInstanceOf[Mnist]
        PrefetchGlue(dut.clockDomain, dut.io.ctrlBus, dut.io.axiMaster,
          () => dut.io.outStream.stream.valid.toBoolean,
          () => dut.io.outStream.stream.ready.toBoolean,
          () => bench.getFloat(dut.io.outStream.stream.payload(0)),
          () => dut.clockDomain.waitSampling(3000))
      },
      MnistReplica.logits _, tag = "BF16")
  }

  test("Mnistw4a8: eager weight prefetch overlaps DDR with compute") {
    val bench = new Mnistw4a8Test
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnistw4a8(axiConfig))
    prefetchBody(
      compiled,
      () => bench.weightWords(),
      (mem, img) => writeWords(mem, imgBase, bench.toWords(bench.imageBytes(img))),
      d => {
        val dut = d.asInstanceOf[Mnistw4a8]
        PrefetchGlue(dut.clockDomain, dut.io.ctrlBus, dut.io.axiMaster,
          () => dut.io.outStream.stream.valid.toBoolean,
          () => dut.io.outStream.stream.ready.toBoolean,
          () => bench.getFloat(dut.io.outStream.stream.payload(0)),
          () => dut.clockDomain.waitSampling(3000))
      },
      Mnistw4a8Replica.logits _, tag = "W4A8")
  }
}
