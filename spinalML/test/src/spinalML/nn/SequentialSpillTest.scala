// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.I8
import spinalML.harness.MemoryHarness
import spinalML.utils.SimLog

/**
 * S2c end-to-end: a spilling Linear (K-pass GEMM) inside a real Sequential +
 * Accelerator, DDR-backed throughout, bit-exact against a hand oracle.
 *
 * Node-0 case (first three tests): M=1, N=4, K=4, Ks=2, P=2, I8,
 * temporal=1. Exclusive node 0, so pass p=1 re-fires the image sweep from
 * DDR (no tap). The final y AND the spill-region content (pass-0 partials,
 * never rewritten by the final pass) are both asserted — together they
 * prove the whole loop: slice fetch, seed=zeros, drain write-back, fence,
 * seed read-back, re-streamed A, single bias.
 *
 * Deep-node case (tap test): [dense 4->4, spill 4->4 Ks=2]. The spill sits
 * at node 1 (A = layer-0 output, 4B I8, within the S0 replay budget), so
 * pass p=1 replays A from the on-chip StreamTap — the image plane stays
 * idle. Layer 0 is a quasi-passthrough (W0 = identity, tiny B0) so the tap
 * replay is the ONLY new variable vs the proven node-0 case: same y oracle
 * shape, same region proof, but A re-streamed on-chip.
 */
class SequentialSpillTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val imgBase = 0x10000L
  val weightBase = 0x40000L
  val spillBase = 0x30000L

  // Distinct values: any lane/beat permutation reads out of the region.
  val AFull = Seq(1, 2, 3, 4)
  val W = (0 until 4).map(k => (0 until 4).map(j => ((k + j) % 3) - 1))
  // Diagnostic bias: huge and distinctive so the EFFECTIVE bias reads out
  // of y-(P0+P1) directly.
  val B = Seq(40, 50, 60, 70)
  // Deep-node (tap) case, layer 0: quasi-passthrough W0 = identity + tiny
  // B0, so y0 = A + B0 stays small and the spill layer-1 oracle below is
  // the only arithmetic under test.
  val W0 = (0 until 4).map(k => (0 until 4).map(j => if (k == j) 1 else 0))
  val B0 = Seq(1, 2, 3, 4)
  // DDR layout contract: each K-slice is stored TRANSPOSED and
  // contiguously (the engine reads B column-major: readAddr = n*chunksK+k,
  // cf. S1 bBeats and the replica's slice(o*K..)). Slice p linear order =
  // for n: for k in slice: W[k][n]. (Legacy whole-transpose would scatter
  // slices strided — unfetchable in one linear DMA.)
  // Full GEMM + bias, exact int math (per-case A).
  def expectedY(a: Seq[Int]) =
    (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W(k)(j) }.sum + B(j))
  // Pass-0 partials (K slice 0..1, no bias).
  def expectedPartials(a: Seq[Int]) =
    (0 until 4).map(j => a(0) * W(0)(j) + a(1) * W(1)(j))

  private def writeWords(mem: spinal.lib.bus.amba4.axi.sim.SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)
  }

  def programmedW(ks: Int): Seq[Int] =
    (0 until 4 by ks).flatMap(p => (0 until 4).flatMap(n => (p until p + ks).map(k => W(k)(n))))

  // Dense (non-spilling) layer layout: full-K column-major, same engine
  // read order as programmedW(4) — one helper per matrix for clarity.
  def programmedFull(m: Seq[Seq[Int]]): Seq[Int] =
    (0 until 4).flatMap(n => (0 until 4).map(k => m(k)(n)))
  // Tap-case oracles: y0 = A*W0+B0 (dense), then the spill layer-1 passes
  // run on y0 exactly like the node-0 case runs on A.
  def expectedY0(a: Seq[Int]) =
    (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W0(k)(j) }.sum + B0(j))
  def expectedY1(y0: Seq[Int]) =
    (0 until 4).map(j => y0.zipWithIndex.map { case (v, k) => v * W(k)(j) }.sum + B(j))
  def expectedPartials1(y0: Seq[Int]) =
    (0 until 4).map(j => y0(0) * W(0)(j) + y0(1) * W(1)(j))

  // S2d scale case: M=1, N=4, K=64, Ks=8, P=8, I8. A is a 0/1 pattern and
  // W the same {-1,0,1} motif so the I8 accumulator never saturates
  // (|y| <= 32+40, |pass partials| <= 4).
  val A64 = (0 until 64).map(k => k % 2)
  val W64 = (0 until 64).map(k => (0 until 4).map(j => ((k + j) % 3) - 1))
  val B64 = Seq(10, 20, 30, 40)
  def programmedW64(ks: Int): Seq[Int] =
    (0 until 64 by ks).flatMap(p => (0 until 4).flatMap(n => (p until p + ks).map(k => W64(k)(n))))
  def expectedY64(a: Seq[Int]) =
    (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W64(k)(j) }.sum + B64(j))
  def expectedPartials64(a: Seq[Int]) =
    (0 until 4).map(j => (0 until 8).map(k => a(k) * W64(k)(j)).sum)

  // S2d discriminator: K=8, Ks=4, P=2. A single re-fire like K=4, but a
  // multi-beat image sweep like K=64. Footprint: image 8B + weights 40B
  // (32B W @0, 4B bias @32) + out 8B + spill 8B = 64B.
  val A8 = (0 until 8).map(k => k % 2)
  val W8 = (0 until 8).map(k => (0 until 4).map(j => ((k + j) % 3) - 1))
  def programmedW8(ks: Int): Seq[Int] =
    (0 until 8 by ks).flatMap(p => (0 until 4).flatMap(n => (p until p + ks).map(k => W8(k)(n))))
  def expectedY8(a: Seq[Int]) =
    (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W8(k)(j) }.sum + B(j))
  def expectedPartials8(a: Seq[Int]) =
    (0 until 4).map(j => (0 until 4).map(k => a(k) * W8(k)(j)).sum)

  def runCase(spec: Seq[LayerSpec], prog: Seq[(Long, Seq[Int])], oracleY: Seq[Int],
      oraclePartials: Option[Seq[Int]], label: String,
      inShape: Seq[Int] = Seq(1, 4), capacity: Option[Long] = Some(128),
      runs: Int = 1): Unit = {
    val compiled = SimConfig.withWave.compile(
      new Accelerator(
        dataType = I8(),
        inputShape = inShape,
        modelSpec = spec,
        axiConfig = axiConfig,
        temporal = 1,
        memory = MemorySpec(spillBase = Some(spillBase), capacityBytes = capacity)
      )
    )
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      def tick(): Unit = {
        val _ = dut.io.outStream.stream.valid.toBoolean
        dut.clockDomain.waitSampling()
      }

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memSim.start()

      // Program DDR: (address, values) pairs built by the caller — image,
      // per-layer W (slice-transposed for a spill layer, full column-major
      // for a dense layer), per-layer bias, at the Sequential region
      // offsets (W+B per layer in order, beat-aligned: see the fetch plane).
      for ((base, values) <- prog) writeWords(memSim.memory, base, MemoryHarness.packBytes(values))

      def writeCsr(addr: BigInt, data: BigInt): Unit = {
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
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      tick(); tick()

      // Bring-up settle: the memory-model agents must own the
      // bus before stimulus, or a spurious B/R wedges a DMA counter.
      var settled = 0
      var sc = 0
      while (settled < 5 && sc < 500) {
        val bLow = !dut.io.axiMaster.b.valid.toBoolean
        val awR = dut.io.axiMaster.aw.ready.toBoolean
        val arR = dut.io.axiMaster.ar.ready.toBoolean
        if (bLow && awR && arR) settled += 1 else settled = 0
        tick(); sc += 1
      }
      assert(settled == 5, "memory-model agents never settled on the bus")

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      // 0x34 spill base defaults to the MemorySpec descriptor (init path
      // under test — no CSR write here).

      def readCsr(addr: BigInt): BigInt = {
        dut.io.ctrlBus.ar.valid #= true
        dut.io.ctrlBus.ar.payload.addr #= addr
        dut.io.ctrlBus.r.ready #= true
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.ar.ready.toBoolean)
        dut.io.ctrlBus.ar.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.r.valid.toBoolean)
        val data = dut.io.ctrlBus.r.payload.data.toBigInt
        dut.io.ctrlBus.r.ready #= false
        dut.clockDomain.waitSampling()
        data
      }

      // S2d-2 run coherence: the same inference re-runs back-to-back (one
      // START per run, DDR left programmed). runs=1 default keeps every
      // existing case single-shot.
      var runTag = label
      for (run <- 1 to runs) {
        if (runs > 1) runTag = s"$label-run$run"
        writeCsr(0x00, 1)

      // Authoritative controller trace (SimLog TRACE, bench reads, not VCD):
      // levels and pulses around both preludes, on change only.
      val ctrl = dut.model.spillCtrl.get
      var lastCtl = ""
      var cycles = 0
      val timeout = 50000
      def traceCtl(): Unit = {
        val s = s"pf=${ctrl.io.passFirst.toBoolean} pl=${ctrl.io.passLast.toBoolean} " +
          s"pi=${ctrl.io.passIdx.toInt} rw=${ctrl.io.refetchW.toBoolean} " +
          s"br=${ctrl.io.biasReArm.toBoolean} ra=${ctrl.io.restartA.toBoolean}"
        if (s != lastCtl) { SimLog.trace("SPILL")(s"ctl [$runTag cyc=$cycles]: $s"); lastCtl = s }
      }

      // Collect up to 8 beats: 4 = silent pass 0, 8 = pass-0
      // leak into y (would explain first-4 anomalies).
      val collected = scala.collection.mutable.ArrayBuffer[Int]()
      // Drain history: sample the spill region every cycle, TRACE on
      // change — shows which pass drains landed and what they wrote.
      var lastMon = memSim.memory.readBigInt(spillBase, 4)
      // Bus monitor: TRACE every AXI read/write command (fire cycle) —
      // W-slice fetch addresses per pass, seed reads, drain writes.
      // Window monitor: sample the K-window position every cycle,
      // TRACE on change — reconstructs exactly which stream beats each
      // pass consumed (winLo = window low beat, aBeat = intra-row beat).
      val spillIdx = dut.model.spillLayerIdx.get
      val winLoSig = dut.model.spillWinLoOf(spillIdx)
      val aBeatSig = dut.model.spillABeatOf(spillIdx)
      var lastWin = (-1, -1)
      while (collected.length < 8 && cycles < timeout) {
        if (dut.io.outStream.stream.valid.toBoolean)
          collected += dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toInt
        if (label != "P=1" && SimLog.isTrace) traceCtl()
        if (SimLog.isTrace) {
          val wab = (winLoSig.toInt, aBeatSig.toInt)
          if (wab != lastWin) {
            SimLog.trace("SPILL")(s"winmon [$runTag cyc=$cycles]: winLo=${wab._1} aBeat=${wab._2} pi=${ctrl.io.passIdx.toInt}")
            lastWin = wab
          }
        }
        if (SimLog.isTrace && dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean) {
          SimLog.trace("SPILL")(s"busmon [$runTag cyc=$cycles]: AR addr=0x${dut.io.axiMaster.ar.payload.addr.toBigInt.toString(16)} len=${dut.io.axiMaster.ar.payload.len.toInt + 1}")
        }
        if (SimLog.isTrace && dut.io.axiMaster.aw.valid.toBoolean && dut.io.axiMaster.aw.ready.toBoolean) {
          SimLog.trace("SPILL")(s"busmon [$runTag cyc=$cycles]: AW addr=0x${dut.io.axiMaster.aw.payload.addr.toBigInt.toString(16)} len=${dut.io.axiMaster.aw.payload.len.toInt + 1}")
        }
        tick(); cycles += 1
        if (SimLog.isTrace && cycles % 4 == 0) {
          val mon = memSim.memory.readBigInt(spillBase, 4)
          if (mon != lastMon) { SimLog.trace("SPILL")(s"spillmon [$runTag cyc=$cycles]: region=0x${mon.toString(16)}"); lastMon = mon }
        }
      }
      println(s"S2c e2e debug [$runTag]: all-y=${collected.toSeq}")
      assert(collected.length == 4, s"expected exactly 4 y beats, got ${collected.length}: ${collected.toSeq}")

      // Spill region: pass-0 partials, verbatim (final pass reads, never
      // rewrites). Signed bytes, little-endian. Read BEFORE asserting y so
      // a failure localizes to pass 0 (region) vs pass 1 (y).
      val region = memSim.memory.readBigInt(spillBase, 4)
      println(s"S2c e2e debug [$runTag]: y=${collected.toSeq} oracleY=$oracleY region=0x${region.toString(16)}")
      oraclePartials.foreach { op =>
        val expectedRegion = op.zipWithIndex.map { case (v, j) =>
          (BigInt(v & 0xFF) << (j * 8))
        }.reduce(_ | _)
        println(s"S2c e2e debug [$runTag]: expectedRegion=0x${expectedRegion.toString(16)}")
        assert(region == expectedRegion,
          f"spill region 0x$region%08X != pass-0 partials 0x$expectedRegion%08X")
      }
      assert(collected.toSeq == oracleY, s"e2e y ${collected.toSeq} != oracle $oracleY")
      tick(); tick()

      // S2d-2 run coherence, read back per run (CsrMap, same package):
      // TILE_CNT counts FRAMES (one multi-pass inference = 1), status is
      // idle after the drain (STOP: no busy, no stray valid), MODE reads 0
      // (STREAM_PER_PASS held for the whole run), and 0x34 still reads the
      // descriptor base (no CSR aliasing by the pass loop).
      val tileCnt = readCsr(CsrMap.TileCnt)
      assert(tileCnt == run, s"TILE_CNT=$tileCnt after run $run (expected $run)")
      val status = readCsr(CsrMap.Status)
      assert(status == 0, s"status=0x${status.toString(16)} after run $run (expected idle STOP)")
      val mode = readCsr(CsrMap.Mode)
      assert(mode == 0, s"MODE=0x${mode.toString(16)} after run $run (expected STREAM_PER_PASS)")
      val spillRb = readCsr(CsrMap.SpillBase)
      assert(spillRb == spillBase, f"CSR 0x34=0x$spillRb%X after run $run (expected 0x$spillBase%X)")
      }
    }
  }

  test("S2c e2e node-0 spill P=2: K-pass GEMM bit-exact, region holds pass-0 partials") {
    runCase(Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 2)),
      Seq(imgBase -> AFull, weightBase -> programmedW(2), (weightBase + 16) -> B),
      expectedY(AFull), Some(expectedPartials(AFull)), label = "P=2")
  }

  test("S2c e2e node-0 spill P=1: single-pass regression (no seed, direct drain)") {
    runCase(Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(imgBase -> AFull, weightBase -> programmedW(4), (weightBase + 16) -> B),
      expectedY(AFull), None, label = "P=1")
  }

  test("S2c e2e node-0 spill P=2 zero-tail A: seed+bias proof (P1 == 0)") {
    runCase(Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 2)),
      Seq(imgBase -> Seq(1, 2, 0, 0), weightBase -> programmedW(2), (weightBase + 16) -> B),
      expectedY(Seq(1, 2, 0, 0)), Some(expectedPartials(Seq(1, 2, 0, 0))), label = "P=2-zero-tail")
  }

  test("S2c e2e deep-node spill P=2 via StreamTap: A replayed on-chip, bit-exact, region holds pass-0 partials") {
    // W/B region offsets for the 2-layer chain (beat = 8B; W region 16B,
    // bias region 4B, each start beat-aligned — mirrors the fetch plane's
    // alignToBeat walk: W0@0, B0@16, W1@24, B1@40).
    def align8(x: Int) = (x + 8 - 1) / 8 * 8
    val w0Off = 0
    val b0Off = align8(w0Off + 16)
    val w1Off = align8(b0Off + 4)
    val b1Off = align8(w1Off + 16)
    val y0 = expectedY0(AFull)
    runCase(
      Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2),
        Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 2)),
      Seq(imgBase -> AFull,
        (weightBase + w0Off) -> programmedFull(W0), (weightBase + b0Off) -> B0,
        (weightBase + w1Off) -> programmedW(2), (weightBase + b1Off) -> B),
      expectedY1(y0), Some(expectedPartials1(y0)), label = "TAP-P2")
  }

  test("S2d e2e node-0 spill K=64 P=8 at exact fit: 8-pass GEMM bit-exact, region holds pass-0 partials") {
    // Footprint: image 64B + weights 264B (256B W @0, 4B
    // bias @256) + out 8B + spill 8B = 344B — the capacity below is exact.
    runCase(Seq(Linear(inFeatures = 64, outFeatures = 4, weightLanes = 2, spillKSlice = 8)),
      Seq(imgBase -> A64, weightBase -> programmedW64(8), (weightBase + 256) -> B64),
      expectedY64(A64), Some(expectedPartials64(A64)), label = "K64-P8",
      inShape = Seq(1, 64), capacity = Some(344))
  }

  test("S2d e2e node-0 spill K=8 P=2: single re-fire, multi-beat sweep") {
    runCase(Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(imgBase -> A8, weightBase -> programmedW8(4), (weightBase + 32) -> B),
      expectedY8(A8), Some(expectedPartials8(A8)), label = "K8-P2",
      inShape = Seq(1, 8), capacity = Some(64))
  }

  test("S2d e2e node-0 spill K=64 P=2: long sweep, single re-fire") {
    // Same 64-elem sweep as K64-P8, one re-fire only. Footprint 344B.
    runCase(Seq(Linear(inFeatures = 64, outFeatures = 4, weightLanes = 2, spillKSlice = 32)),
      Seq(imgBase -> A64, weightBase -> programmedW64(32), (weightBase + 256) -> B64),
      expectedY64(A64), Some(expectedPartials64(A64)), label = "K64-P2",
      inShape = Seq(1, 64), capacity = Some(344))
  }

  test("S2d e2e node-0 spill K=12 P=3: short sweep, three passes") {
    // 12-elem sweep (like K8, known-clean re-fire) x 3 passes (like K64).
    // Footprint: image 12B + weights 56B (48B W @0, 4B bias @48) + out 8B
    // + spill 8B = 84B.
    val A12 = (0 until 12).map(k => k % 2)
    val W12 = (0 until 12).map(k => (0 until 4).map(j => ((k + j) % 3) - 1))
    def programmedW12(ks: Int): Seq[Int] =
      (0 until 12 by ks).flatMap(p => (0 until 4).flatMap(n => (p until p + ks).map(k => W12(k)(n))))
    def expectedY12(a: Seq[Int]) =
      (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W12(k)(j) }.sum + B(j))
    def expectedPartials12(a: Seq[Int]) =
      (0 until 4).map(j => (0 until 4).map(k => a(k) * W12(k)(j)).sum)
    // S2d region invariant: the final pass drains to y, never to DDR, so
    // the region holds drain_{P-2} = the unbiased cumulative through the
    // last NON-final pass (P=2: P0; here P=3: P0+P1).
    def expectedCumulative12(a: Seq[Int]) =
      (0 until 4).map(j => (0 until 8).map(k => a(k) * W12(k)(j)).sum)
    runCase(Seq(Linear(inFeatures = 12, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(imgBase -> A12, weightBase -> programmedW12(4), (weightBase + 48) -> B),
      expectedY12(A12), Some(expectedCumulative12(A12)), label = "K12-P3",
      inShape = Seq(1, 12), capacity = Some(84))
  }

  test("S2d e2e node-0 spill K=16 P=4 self-identifying: every pass recognizable") {
    // A[k]=k+1, W[k][j]=1 iff k%4==j: pass p computes Pp=(4p+1,...,4p+4),
    // so any slice/window/shift error reads out unambiguously. I8-safe:
    // |y| <= 110, |Pp| <= 64. Region = drain_{P-2} = P0+P1+P2.
    // Footprint: image 16B + weights 72B (64B W @0, 4B bias @64, beat-
    // aligned) + out 8B + spill 8B = 104B.
    val A16 = (0 until 16).map(k => k + 1)
    val W16 = (0 until 16).map(k => (0 until 4).map(j => if (k % 4 == j) 1 else 0))
    def programmedW16(ks: Int): Seq[Int] =
      (0 until 16 by ks).flatMap(p => (0 until 4).flatMap(n => (p until p + ks).map(k => W16(k)(n))))
    def expectedY16(a: Seq[Int]) =
      (0 until 4).map(j => a.zipWithIndex.map { case (av, k) => av * W16(k)(j) }.sum + B(j))
    def expectedPass16(a: Seq[Int], p: Int) =
      (0 until 4).map(j => (4 * p until 4 * p + 4).map(k => a(k) * W16(k)(j)).sum)
    def expectedCumulative16(a: Seq[Int]) =
      (0 until 4).map(j => (0 until 12).map(k => a(k) * W16(k)(j)).sum)
    runCase(Seq(Linear(inFeatures = 16, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(imgBase -> A16, weightBase -> programmedW16(4), (weightBase + 64) -> B),
      expectedY16(A16), Some(expectedCumulative16(A16)), label = "K16-P4",
      inShape = Seq(1, 16), capacity = Some(104))
  }

  test("S2d run coherence: back-to-back rerun reproduces y, TILE_CNT/STOP/cursors coherent") {
    // Same P=2 node-0 model twice on one programmed DDR: y and region
    // identical both runs, TILE_CNT counts frames (multi-pass = 1).
    runCase(Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 2)),
      Seq(imgBase -> AFull, weightBase -> programmedW(2), (weightBase + 16) -> B),
      expectedY(AFull), Some(expectedPartials(AFull)), label = "RERUN",
      runs = 2)
  }
}
