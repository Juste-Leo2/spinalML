// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.I8
import spinalML.harness.MemoryHarness

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
  // of y-(P0+P1) directly (S2c e2e debug).
  val B = Seq(40, 50, 60, 70)
  // Deep-node (tap) case, layer 0: quasi-passthrough W0 = identity + tiny
  // B0, so y0 = A + B0 stays small and the spill layer-1 oracle below is
  // the only arithmetic under test.
  val W0 = (0 until 4).map(k => (0 until 4).map(j => if (k == j) 1 else 0))
  val B0 = Seq(1, 2, 3, 4)
  // DDR layout contract (S2): each K-slice is stored TRANSPOSED and
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

  def runCase(spec: Seq[LayerSpec], prog: Seq[(Long, Seq[Int])], oracleY: Seq[Int],
      oraclePartials: Option[Seq[Int]], label: String): Unit = {
    val compiled = SimConfig.withWave.compile(
      new Accelerator(
        dataType = I8(),
        inputShape = Seq(1, 4),
        modelSpec = spec,
        axiConfig = axiConfig,
        temporal = 1,
        memory = MemorySpec(spillBase = Some(spillBase), capacityBytes = Some(128))
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

      // Bring-up settle (S2b lesson): the memory-model agents must own the
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
      writeCsr(0x00, 1)

      // Authoritative controller trace (bench reads, not VCD): levels and
      // pulses around both preludes. Prints on change only.
      val ctrl = dut.model.spillCtrl.get
      var lastCtl = ""
      def traceCtl(): Unit = {
        val s = s"pf=${ctrl.io.passFirst.toBoolean} pl=${ctrl.io.passLast.toBoolean} " +
          s"pi=${ctrl.io.passIdx.toInt} rw=${ctrl.io.refetchW.toBoolean} " +
          s"br=${ctrl.io.biasReArm.toBoolean} ra=${ctrl.io.restartA.toBoolean}"
        if (s != lastCtl) { println(s"S2c ctl: $s"); lastCtl = s }
      }

      // Collect up to 8 beats

      // Collect up to 8 beats: 4 = silent pass 0 (S1 contract), 8 = pass-0
      // leak into y (would explain first-4 anomalies).
      val collected = scala.collection.mutable.ArrayBuffer[Int]()
      var cycles = 0
      val timeout = 50000
      while (collected.length < 8 && cycles < timeout) {
        if (dut.io.outStream.stream.valid.toBoolean)
          collected += dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toInt
        if (label != "P=1") traceCtl()
        tick(); cycles += 1
      }
      println(s"S2c e2e debug [$label]: all-y=${collected.toSeq}")
      assert(collected.length == 4, s"expected exactly 4 y beats, got ${collected.length}: ${collected.toSeq}")

      // Spill region: pass-0 partials, verbatim (final pass reads, never
      // rewrites). Signed bytes, little-endian. Read BEFORE asserting y so
      // a failure localizes to pass 0 (region) vs pass 1 (y).
      val region = memSim.memory.readBigInt(spillBase, 4)
      println(s"S2c e2e debug [$label]: y=${collected.toSeq} oracleY=$oracleY region=0x${region.toString(16)}")
      oraclePartials.foreach { op =>
        val expectedRegion = op.zipWithIndex.map { case (v, j) =>
          (BigInt(v & 0xFF) << (j * 8))
        }.reduce(_ | _)
        println(s"S2c e2e debug [$label]: expectedRegion=0x${expectedRegion.toString(16)}")
        assert(region == expectedRegion,
          f"spill region 0x$region%08X != pass-0 partials 0x$expectedRegion%08X")
      }
      assert(collected.toSeq == oracleY, s"e2e y ${collected.toSeq} != oracle $oracleY")
      tick(); tick()
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
}
