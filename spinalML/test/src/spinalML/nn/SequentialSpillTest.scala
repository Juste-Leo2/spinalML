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
 * Node-0 case (this file, first test): M=1, N=4, K=4, Ks=2, P=2, I8,
 * temporal=1. Exclusive node 0, so pass p=1 re-fires the image sweep from
 * DDR (no tap). The final y AND the spill-region content (pass-0 partials,
 * never rewritten by the final pass) are both asserted — together they
 * prove the whole loop: slice fetch, seed=zeros, drain write-back, fence,
 * seed read-back, re-streamed A, single bias.
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

  def runCase(a: Seq[Int], spillKSlice: Int, progW: Seq[Int], checkRegion: Boolean, label: String): Unit = {
    val oracleY = expectedY(a)
    val oraclePartials = expectedPartials(a)
    val spec = Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = spillKSlice))
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

      // Program DDR: image, W per-slice-transposed (see contract above), bias.
      writeWords(memSim.memory, imgBase, MemoryHarness.packBytes(a))
      writeWords(memSim.memory, weightBase, MemoryHarness.packBytes(progW))
      writeWords(memSim.memory, weightBase + 16, MemoryHarness.packBytes(B))

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
      val expectedRegion = oraclePartials.zipWithIndex.map { case (v, j) =>
        (BigInt(v & 0xFF) << (j * 8))
      }.reduce(_ | _)
      println(s"S2c e2e debug [$label]: y=${collected.toSeq} oracleY=$oracleY region=0x${region.toString(16)} expectedRegion=0x${expectedRegion.toString(16)}")
      if (checkRegion) assert(region == expectedRegion,
        f"spill region 0x$region%08X != pass-0 partials 0x$expectedRegion%08X")
      assert(collected.toSeq == oracleY, s"e2e y ${collected.toSeq} != oracle $oracleY")
      tick(); tick()
    }
  }

  test("S2c e2e node-0 spill P=2: K-pass GEMM bit-exact, region holds pass-0 partials") {
    runCase(AFull, 2, programmedW(2), checkRegion = true, label = "P=2")
  }

  test("S2c e2e node-0 spill P=1: single-pass regression (no seed, direct drain)") {
    runCase(AFull, 4, programmedW(4), checkRegion = false, label = "P=1")
  }

  test("S2c e2e node-0 spill P=2 zero-tail A: seed+bias proof (P1 == 0)") {
    runCase(Seq(1, 2, 0, 0), 2, programmedW(2), checkRegion = true, label = "P=2-zero-tail")
  }
}
