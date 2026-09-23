// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinal.lib.sim.{StreamDriver, StreamMonitor}
import spinalML.dtypes.I8
import spinalML.tensors.Tensor
import spinalML.memory.{DMAReader, DMAWriter}
import org.scalatest.funsuite.AnyFunSuite

// S2b unit harness: SpillPassController + spill DMA pair against AxiMemorySim.
// P=2, I8 on a 64-bit AXI bus. The bench stubs the compute side: it sources
// the drain stream, sinks the seed stream, and scripts passDone / wFetchFire.
// Pass 1 must read back exactly what pass 0 wrote (single-region loopback
// through DDR). Default M=1, N=4: 4 partial elems = 1 AXI beat.
case class SpillPassLoopWrapper(M: Int = 1, N: Int = 4) extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val accType = I8()
  // Region beats covering the M*N I8 partials (ceil to whole beats; the last
  // beat's pad is memory fill — the stub bench only collects M*N).
  val spillBeats = (M * N + 7) / 8

  val io = new Bundle {
    val start = in Bool()
    val passDone = in Bool()
    val wFetchFire = in Bool()
    val residentMode = in Bool()
    val spillBase = in UInt(32 bits)
    // Compute-side spill streams (stubbed by the bench).
    val seedOut = master(Tensor(accType, Seq(M, N), lanes = 1))
    val drainIn = slave(Tensor(accType, Seq(M, N), lanes = 1))
    // Controller observables (the S2b contract under test).
    val passFirst = out Bool()
    val passLast = out Bool()
    val passIdx = out UInt(1 bits)
    val refetchW = out Bool()
    val biasReArm = out Bool()
    val busy = out Bool()
    val done = out Bool()
    val readerCmdFire = out Bool()
    val readerCmdAddr = out UInt(32 bits)
    val readerCmdLen = out UInt(16 bits)
    val writerCmdFire = out Bool()
    val writerCmdAddr = out UInt(32 bits)
    val writerCmdLen = out UInt(16 bits)
    val writerDone = out Bool()
    val axiMaster = master(Axi4(axiConfig))
  }

  val ctrl = SpillPassController(passes = 2, addrWidth = 32, spillBeats = spillBeats)
  ctrl.io.start := io.start
  ctrl.io.spillBase := io.spillBase
  ctrl.io.passDone := io.passDone
  ctrl.io.wFetchFire := io.wFetchFire
  ctrl.io.residentMode := io.residentMode

  val reader = DMAReader(accType, Seq(M, N), outLanes = 1, axiConfig,
    // S2e production recipe mirror: single-beat seed chunks, so no trim (the
    // trim counter restarts at every cmd.fire); the flushable accept gate
    // paces chunks at the stub's consumption rate. No engine pad-drain here
    // (the stub consumes exactly what the test collects).
    trimToElements = false, flushableGearbox = true)
  val writer = DMAWriter(accType, Seq(M, N), inLanes = 1, axiConfig)
  reader.io.cmd << ctrl.io.readerCmd
  writer.io.cmd << ctrl.io.writerCmd
  ctrl.io.writerDone := writer.io.done
  io.seedOut.stream << reader.io.outStream.stream
  writer.io.inStream.stream << io.drainIn.stream

  io.passFirst := ctrl.io.passFirst
  io.passLast := ctrl.io.passLast
  io.passIdx := ctrl.io.passIdx
  io.refetchW := ctrl.io.refetchW
  io.biasReArm := ctrl.io.biasReArm
  io.busy := ctrl.io.busy
  io.done := ctrl.io.done
  io.readerCmdFire := ctrl.io.readerCmd.fire
  io.readerCmdAddr := ctrl.io.readerCmd.address
  io.readerCmdLen := ctrl.io.readerCmd.length
  io.writerCmdFire := ctrl.io.writerCmd.fire
  io.writerCmdAddr := ctrl.io.writerCmd.address
  io.writerCmdLen := ctrl.io.writerCmd.length
  io.writerDone := writer.io.done

  // Read side: reader only. Write side: writer only.
  io.axiMaster.ar << reader.io.axiMaster.ar
  reader.io.axiMaster.r << io.axiMaster.r
  io.axiMaster.aw << writer.io.axiMaster.aw
  io.axiMaster.w << writer.io.axiMaster.w
  writer.io.axiMaster.b << io.axiMaster.b
}

class SpillPassControllerTest extends AnyFunSuite {

  // S1 bench discipline, controller-scale: every edge is preceded by a DUT
  // read (poke-then-edge races the clock), valids are held (never toggled
  // per beat), payloads pre-poked, exactly one tick per beat.
  test("S2b pass loop: write-back, fence, seed read-back, done") {
    SimConfig.withWave.compile {
      val dut = SpillPassLoopWrapper()
      dut.setDefinitionName("SpillPassLoopComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.busy.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.start #= false
      dut.io.passDone #= false
      dut.io.wFetchFire #= false
      dut.io.residentMode #= false
      dut.io.spillBase #= 0x1000
      dut.io.drainIn.stream.valid #= false
      dut.io.seedOut.stream.ready #= true
      tick(); tick()

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4))
      memSim.start()
      tick(); tick()

      // Settle: the memory-model agents own the bus only once their threads
      // poke the slave idles. Stimulating earlier races agent bring-up — a
      // B-high transient then wedges DMAWriter.pendingB for the whole run
      // (b.ready is hardwired True, so one spurious B underflows 0 -> FF and
      // cmd.ready sticks low forever). Poll the bus, not luck.
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

      // ---- Pass 0 prelude ------------------------------------------------
      // Single observation loop: writerCmd.fire is a single cycle issued on
      // prelude entry — polling for it AFTER watching bias would miss it
      // (same record-vs-transfer trap as S1). Observe everything together.
      dut.io.start #= true
      tick()
      dut.io.start #= false
      var sawBusy = false
      var sawBias = false
      var sawWriterCmd = false
      var sawReaderCmd = false
      var wAddr = 0L
      var wLen = -1
      var biasPulses = 0
      var cycles = 0
      while (!(sawBusy && sawBias && sawWriterCmd) && cycles < 100) {
        if (dut.io.busy.toBoolean) sawBusy = true
        if (dut.io.biasReArm.toBoolean) { biasPulses += 1; sawBias = true }
        if (dut.io.writerCmdFire.toBoolean) {
          sawWriterCmd = true
          wAddr = dut.io.writerCmdAddr.toLong
          wLen = dut.io.writerCmdLen.toInt
        }
        if (dut.io.readerCmdFire.toBoolean) sawReaderCmd = true
        tick(); cycles += 1
      }
      assert(sawBusy, "controller never went busy after START")
      assert(sawBias, "biasReArm never pulsed in pass-0 prelude")
      assert(sawWriterCmd, "pass-0 drain write never commanded")
      // Controller busy, pass-0 levels, index 0.
      assert(dut.io.passFirst.toBoolean, "pass 0 must hold passFirst")
      assert(!dut.io.passLast.toBoolean, "pass 0 of P=2 must not hold passLast")
      assert(dut.io.passIdx.toInt == 0, "pass index must be 0")
      // No seed read on the first pass (zeros come from the compute side).
      assert(!sawReaderCmd, "pass 0 must not command a seed read")
      // ...but the drain write IS commanded, at the region base, 1 beat.
      assert(wAddr == 0x1000, f"writer addr 0x$wAddr%X != spill base 0x1000")
      assert(wLen == 0, "1 AXI beat => length 0")

      // ---- Pass 0 drain: source 4 partial beats ---------------------------
      val partials = Seq(1, 2, 3, 4)
      dut.io.drainIn.stream.payload(0) #= partials(0)
      tick()
      dut.io.drainIn.stream.valid #= true
      for (i <- partials.indices) {
        tick()
        if (i + 1 < partials.length) dut.io.drainIn.stream.payload(0) #= partials(i + 1)
      }
      dut.io.drainIn.stream.valid #= false
      // Compute signals drain completion.
      dut.io.passDone #= true
      tick()
      dut.io.passDone #= false
      // Fence: the controller advances only after writerDone (same-address
      // RAW on the single region).
      cycles = 0
      while (!dut.io.writerDone.toBoolean && cycles < 200) { tick(); cycles += 1 }
      assert(dut.io.writerDone.toBoolean, "writerDone never pulsed after pass-0 drain")
      // Seed collector, declared before the advance poll: seed beats can
      // flow as soon as the pass-1 prelude fires the read, so every wait
      // loop from here on collects (never collect-after-wait on a live
      // stream).
      val seed = scala.collection.mutable.ArrayBuffer[Int]()
      def collectSeed(): Unit = {
        if (dut.io.seedOut.stream.valid.toBoolean)
          seed += dut.io.seedOut.stream.payload(0).toInt
      }
      cycles = 0
      while (dut.io.passIdx.toInt != 1 && cycles < 100) {
        collectSeed()
        tick(); cycles += 1
      }
      assert(dut.io.passIdx.toInt == 1, "controller never advanced to pass 1")
      // Catch the pass-1 entry pulse: the 1-cycle biasReArm fires on prelude
      // entry (~1 cycle after the index flips), so observe IMMEDIATELY — any
      // settle ticks first would consume it (same record-vs-transfer trap).
      // Levels (regs) settle right after; assert them once the pulse is seen.
      var sawBias1 = false
      var sawReader = false
      var rAddr = 0L
      var rLen = -1
      // Spurious-R tripwire: an R beat with no AR outstanding would wrap the
      // reader's burstRemain (same wedge class as the writer pendingB) — the
      // memory model must never do this; record it if it does.
      var spuriousR = 0
      cycles = 0
      while (!(sawBias1 && sawReader) && cycles < 150) {
        if (dut.io.biasReArm.toBoolean) { biasPulses += 1; sawBias1 = true }
        if (dut.io.readerCmdFire.toBoolean) {
          sawReader = true
          rAddr = dut.io.readerCmdAddr.toLong
          rLen = dut.io.readerCmdLen.toInt
        }
        if (dut.io.axiMaster.r.valid.toBoolean && !dut.io.axiMaster.ar.valid.toBoolean)
          spuriousR += 1
        collectSeed()
        tick(); cycles += 1
      }
      assert(sawBias1, "biasReArm never pulsed in pass-1 prelude")
      assert(biasPulses == 2, s"expected exactly 2 bias pulses, saw $biasPulses")
      // Seed read commanded at the same region base, 1 beat.
      assert(sawReader, s"pass-1 seed read never commanded (spuriousR=$spuriousR)")
      assert(rAddr == 0x1000, f"reader addr 0x$rAddr%X != spill base 0x1000")
      assert(rLen == 0, "1 AXI beat => length 0")
      tick()
      assert(!dut.io.passFirst.toBoolean, "pass 1 must not hold passFirst")
      assert(dut.io.passLast.toBoolean, "final pass must hold passLast")

      // ---- Pass 1 prelude: refetch ack + seed command -----------------------
      // Bias pulse and levels already observed at advance time above; here
      // only the sticky servants (refetchW, reader cmd) remain — no race.
      // ---- Pass 1 seed: the early beats were already collected ------------
      // alongside the prelude waits above; drain the tail here.
      var sawRefetch = false
      var refetchAcked = false
      var ackHigh = false
      cycles = 0
      while (!(sawRefetch && refetchAcked) && cycles < 150) {
        if (dut.io.refetchW.toBoolean) sawRefetch = true
        if (sawRefetch && !refetchAcked) {
          if (!ackHigh) { dut.io.wFetchFire #= true; ackHigh = true }
          else { dut.io.wFetchFire #= false; refetchAcked = true }
        }
        collectSeed()
        tick(); cycles += 1
      }
      assert(sawRefetch, "refetchW never asserted for pass 1")
      // W-slice refetch released after the fetch ack.
      cycles = 0
      while (dut.io.refetchW.toBoolean && cycles < 50) {
        collectSeed()
        tick(); cycles += 1
      }
      assert(!dut.io.refetchW.toBoolean, "refetchW never released after fetch ack")
      // Drain the tail.
      cycles = 0
      while (seed.length < 4 && cycles < 200) {
        collectSeed()
        tick(); cycles += 1
      }
      assert(seed.toSeq == partials, s"seed read-back ${seed.toSeq} != written $partials")
      // Final pass commands NO drain write.
      var waited = 0
      var writerFiredFinal = false
      while (waited < 20) {
        if (dut.io.writerCmdFire.toBoolean) writerFiredFinal = true
        tick(); waited += 1
      }
      assert(!writerFiredFinal, "final pass must not command a drain write")

      // ---- Finish ------------------------------------------------------------
      dut.io.passDone #= true
      tick()
      dut.io.passDone #= false
      cycles = 0
      while (!dut.io.done.toBoolean && cycles < 50) { tick(); cycles += 1 }
      assert(dut.io.done.toBoolean, "controller done never pulsed after final pass")
      tick(); tick()
      assert(!dut.io.busy.toBoolean, "controller still busy after done")

      // ---- DDR proof: the region holds exactly the pass-0 partials -----------
      // 4 I8 lanes packed low; the final-beat strobes mask the padding, so
      // only the owned 4 bytes are compared (upper bytes keep whatever the
      // memory-model default fill is — out of the transfer's contract).
      val memWord = memSim.memory.readBigInt(0x1000, 4)
      val expected = BigInt(0x04030201L)
      assert(memWord == expected, f"DDR region 0x$memWord%08X != 0x$expected%08X")
    }
  }

  test("S2b multi-beat seed chunking: paced loopback over 2 beats") {
    // S2e pin: M=2, N=8 I8 = 16 partials = 2 AXI beats (exact, no pad). The
    // pass-1 seed must arrive as TWO single-beat chunks at base/base+8
    // (length 0 each), paced by the reader's accept gate, and read back
    // exactly what pass 0 drained. The drain write stays one full-region
    // command (length 1).
    SimConfig.withWave.compile {
      val dut = SpillPassLoopWrapper(M = 2, N = 8)
      dut.setDefinitionName("SpillPassLoopChunkComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.busy.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.start #= false
      dut.io.passDone #= false
      dut.io.wFetchFire #= false
      dut.io.residentMode #= false
      dut.io.spillBase #= 0x1000
      dut.io.seedOut.stream.ready #= true
      tick(); tick()

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4))
      memSim.start()
      tick(); tick()

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

      val partials = (1 to 16).toSeq
      // Race-free scripting (manual ready/valid sampling races the sim
      // kernel at backpressure transients — single-beat flakiness at the
      // beat boundary). The framework's onSamplings driver/monitor own the
      // handshakes kernel-synchronized: the drain queue feeds exactly 16
      // beats (backpressure-safe), the always-ready seed sink collects each
      // valid beat exactly once.
      val seed = scala.collection.mutable.ArrayBuffer[Int]()
      StreamMonitor(dut.io.seedOut.stream, dut.clockDomain) { p => seed += p(0).toInt }
      val (_, drainQueue) = StreamDriver.queue(dut.io.drainIn.stream, dut.clockDomain)
      partials.foreach { v => drainQueue.enqueue { p => p(0) #= v } }

      // ---- Pass 0: full-region drain write (length 1 = 2 beats) -----------
      dut.io.start #= true
      tick()
      dut.io.start #= false
      var sawWriterCmd = false
      var wLen = -1
      var cycles = 0
      while (!sawWriterCmd && cycles < 100) {
        if (dut.io.writerCmdFire.toBoolean) {
          sawWriterCmd = true
          wLen = dut.io.writerCmdLen.toInt
        }
        tick(); cycles += 1
      }
      assert(sawWriterCmd, "pass-0 drain write never commanded")
      assert(wLen == 1, s"drain write length $wLen != 1 (2-beat region)")
      // Bench-side sticky done (mirror of the DUT's writerDoneSeen latch):
      // the 2-beat drain can complete while the queue is still pushing, i.e.
      // before passDone — a raw-pulse poll would miss the early done
      // exactly like the pre-S2e fence did. writerDone is a clocked-reg
      // pulse, so every-cycle sampling catches it.
      var wDoneSeen = false
      def pollDone(): Unit = {
        if (dut.io.writerDone.toBoolean) wDoneSeen = true
      }
      cycles = 0
      while (!wDoneSeen && cycles < 600) {
        pollDone()
        tick(); cycles += 1
      }
      assert(wDoneSeen, "writerDone never pulsed after pass-0 drain")
      // Compute signals drain completion (the queue is fully accepted once
      // the writer is done — 16 in, 16 packed, 2 beats out).
      dut.io.passDone #= true
      tick()
      dut.io.passDone #= false
      // Seed-chunk record starts here: chunk 0 can fire as soon as the
      // pass-1 prelude opens (same record-vs-transfer trap as S1) — the
      // passIdx poll below must record fires, not just collect seeds.
      val chunkAddrs = scala.collection.mutable.ArrayBuffer[Long]()
      val chunkLens = scala.collection.mutable.ArrayBuffer[Int]()
      def recordChunk(): Unit = {
        if (dut.io.readerCmdFire.toBoolean) {
          chunkAddrs += dut.io.readerCmdAddr.toLong
          chunkLens += dut.io.readerCmdLen.toInt
        }
      }
      cycles = 0
      while (dut.io.passIdx.toInt != 1 && cycles < 100) {
        recordChunk()
        tick(); cycles += 1
      }
      assert(dut.io.passIdx.toInt == 1, "controller never advanced to pass 1")

      // ---- Pass 1: two single-beat seed chunks, then refetch ack -----------
      var sawRefetch = false
      var refetchAcked = false
      var ackHigh = false
      cycles = 0
      while (!(chunkAddrs.length == 2 && sawRefetch && refetchAcked) && cycles < 300) {
        recordChunk()
        if (dut.io.refetchW.toBoolean) sawRefetch = true
        if (sawRefetch && !refetchAcked) {
          if (!ackHigh) { dut.io.wFetchFire #= true; ackHigh = true }
          else { dut.io.wFetchFire #= false; refetchAcked = true }
        }
        tick(); cycles += 1
      }
      assert(chunkAddrs.length == 2, s"expected 2 seed chunks, saw ${chunkAddrs.length}")
      assert(chunkAddrs.toSeq == Seq(0x1000L, 0x1008L),
        s"seed chunk addrs ${chunkAddrs.map(a => f"0x$a%X")} != [0x1000, 0x1008]")
      assert(chunkLens.forall(_ == 0), s"seed chunk lens $chunkLens != [0, 0]")
      assert(sawRefetch, "refetchW never asserted for pass 1")
      // The monitor collected seed beats alongside every wait above; drain
      // the tail here.
      cycles = 0
      while (seed.length < 16 && cycles < 300) {
        tick(); cycles += 1
      }
      assert(seed.toSeq == partials, s"seed read-back ${seed.toSeq} != written $partials")

      // ---- Finish ------------------------------------------------------------
      dut.io.passDone #= true
      tick()
      dut.io.passDone #= false
      cycles = 0
      while (!dut.io.done.toBoolean && cycles < 50) { tick(); cycles += 1 }
      assert(dut.io.done.toBoolean, "controller done never pulsed after final pass")
      tick(); tick()
      assert(!dut.io.busy.toBoolean, "controller still busy after done")

      // ---- DDR proof: both beats hold the pass-0 partials --------------------
      val lo = memSim.memory.readBigInt(0x1000, 8)
      val hi = memSim.memory.readBigInt(0x1008, 8)
      assert(lo == BigInt("0807060504030201", 16), f"DDR beat0 0x$lo%016X mismatch")
      assert(hi == BigInt("100F0E0D0C0B0A09", 16), f"DDR beat1 0x$hi%016X mismatch")
    }
  }

  test("S2b fail-safe: refetch suppressed under residency (loud stall)") {
    SimConfig.withWave.compile {
      val dut = SpillPassLoopWrapper()
      dut.setDefinitionName("SpillPassLoopResidencyComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.busy.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.start #= false
      dut.io.passDone #= false
      dut.io.wFetchFire #= false
      dut.io.residentMode #= true // contract violation, scripted
      dut.io.spillBase #= 0x1000
      dut.io.drainIn.stream.valid #= false
      dut.io.seedOut.stream.ready #= true
      tick(); tick()

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4))
      memSim.start()
      tick(); tick()

      // Same bring-up settle as the main test (spurious-B wedge, see above).
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

      dut.io.start #= true
      tick()
      dut.io.start #= false
      var cycles = 0
      while (!dut.io.busy.toBoolean && cycles < 50) { tick(); cycles += 1 }
      assert(dut.io.busy.toBoolean, "controller never went busy")

      // Run pass 0 to completion (drain + fence need real beats).
      cycles = 0
      while (!dut.io.writerCmdFire.toBoolean && cycles < 50) { tick(); cycles += 1 }
      assert(dut.io.writerCmdFire.toBoolean, "pass-0 drain write never commanded")
      dut.io.drainIn.stream.payload(0) #= 7
      tick()
      dut.io.drainIn.stream.valid #= true
      for (_ <- 0 until 4) tick()
      dut.io.drainIn.stream.valid #= false
      dut.io.passDone #= true
      tick()
      dut.io.passDone #= false
      cycles = 0
      while (dut.io.passIdx.toInt != 1 && cycles < 200) { tick(); cycles += 1 }
      assert(dut.io.passIdx.toInt == 1, "never reached pass 1")

      // Pass-1 prelude under residency: refetchW must NEVER assert (the
      // STREAM_PER_PASS contract), so the loop stalls loudly instead of
      // running a pass on a stale slice.
      var waited = 0
      var refetchSeen = false
      while (waited < 60) {
        if (dut.io.refetchW.toBoolean) refetchSeen = true
        tick(); waited += 1
      }
      assert(!refetchSeen, "refetchW asserted under residency — fail-safe broken")
      assert(dut.io.busy.toBoolean, "controller should still be busy (stalled, loud)")
    }
  }
}
