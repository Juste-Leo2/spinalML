// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.memory.{FetchRequest, WriteRequest}

/**
 * Spill pass controller: owns the K-pass loop of the single v1 spilling layer.
 *
 * One pass `p` on `P` slices:
 *   1. Prelude — hold `passFirst`/`passLast` for the pass; fire the drain
 *      write (p < P-1) against the single M*N spill region; re-fire the
 *      W-slice fetch (p > 0, held until the fetch plane accepts — sticky
 *      discipline); pulse `biasReArm` once (the bias stays parked
 *      from the START fetch, consumed final only). The seed read (p > 0) is
 *      issued as `spillBeats` SINGLE-BEAT chunks (see below), starting
 *      in the prelude and continuing into WaitPass as the engine consumes.
 *   2. WaitPass — the compute drains (seed self-synchronizes by
 *      backpressure: each chunk is accepted only once the previous one was
 *      fully received AND consumed, so a slow seed only stalls, never
 *      deadlocks).
 *   3. WaitFence (non-final) — the next pass starts only after
 *      `writerDone`: same-address RAW on the single region. `writerDone` is
 *      latched (sticky discipline): the 1-cycle pulse may arrive
 *      BEFORE `passDone` (fast memory completing the drain while the engine
 *      still drop-drains the seed pad), and a pre-fence pulse must never be
 *      missed.
 *
 * Single-beat seed chunks (conv P=2 deadlock fix): the seed used to be
 * one full-region command. Past the seed reader's 1-beat flushable gearbox
 * the engine only takes N beats per row (row-interleaved seeding), so
 * an in-order memory wedged: unconsumed seed beats ahead of the A (image)
 * beats the engine waited for, while the seed waited for the engine — a
 * circular wait. Chunking bounds the in-flight seed to one beat: a chunk is
 * accepted only when the reader's gearbox is empty (received AND consumed),
 * which is exactly the engine's consumption pace. Every R stall therefore
 * resolves through engine consumption needing no bus data, so image/W
 * beats behind always flow. The engine drop-drains the last beat's pad
 * elements before passDone (spillPadElems), keeping the gate truthful
 * across passes.
 *
 * P == 1 (full-width single pass) flows through with no
 * reader/writer command: seed zeros, drain straight to `io.y`.
 *
 * STREAM_PER_PASS fail-safe: `refetchW` is suppressed under residency —
 * a spill under resident/prefetch modes stalls loudly instead of silently
 * corrupting (see the runtime-contract note in Sequential).
 */
case class SpillPassController(
  passes: Int,
  addrWidth: Int = 32,
  spillBeats: Int,
  // AXI beat size in bytes (seed-chunk address stride). Must match the bus
  // the seed reader sits on (Sequential passes axiConfig.dataWidth / 8).
  beatBytes: Int = 8
) extends Component {
  require(passes >= 1, s"SpillPassController passes=$passes must be >= 1")
  require(spillBeats >= 1 && spillBeats <= 65536,
    s"SpillPassController spillBeats=$spillBeats must fit one DMA command (1..65536)")

  object State extends SpinalEnum {
    val sIdle, sPrelude, sWaitPass, sWaitFence, sFinish = newElement()
  }

  val io = new Bundle {
    val start = in Bool() // command-boundary pulse (START rising edge)
    val spillBase = in UInt(addrWidth bits) // spill region base (CSR 0x34 + cursor)
    val passDone = in Bool() // compute pass-drain completion pulse
    val writerDone = in Bool() // spill DMAWriter.done pulse
    val wFetchFire = in Bool() // W-slice fetch accepted (reqW.fire of the layer)
    val residentMode = in Bool() // residency control plane level (fail-safe)

    val passFirst = out Bool() // stable per pass
    val passLast = out Bool() // stable per pass
    val passIdx = out UInt((log2Up(passes) max 1) bits) // current pass (fetch addressing)
    val refetchW = out Bool() // re-fire the W-slice fetch (held till wFetchFire)
    val biasReArm = out Bool() // one-cycle pulse per pass prelude
    // One-cycle A re-stream pulse per pass p > 0 (prelude). Sequential
    // routes it: exclusive node-0 spill restarts the image band sweep
    // (DDR-backed, free); shared/deep nodes replay the StreamTap. Suppressed
    // under residency with refetchW (same STREAM_PER_PASS fail-safe).
    val restartA = out Bool()
    val readerCmd = master(Stream(FetchRequest(addrWidth)))
    val writerCmd = master(Stream(WriteRequest(addrWidth)))
    val busy = out Bool()
    val done = out Bool() // one-cycle pulse when the final pass drains
  }
  // Test-only observability: bench-side reads of the pass levels/pulses.
  // Zero behavior change (test-only visibility into an internal controller).
  io.passFirst.simPublic()
  io.passLast.simPublic()
  io.passIdx.simPublic()
  io.refetchW.simPublic()
  io.biasReArm.simPublic()
  io.restartA.simPublic()
  io.busy.simPublic()
  io.done.simPublic()

  val state = RegInit(State.sIdle)
  val prevState = RegInit(State.sIdle)
  prevState := state
  val cnt = Reg(UInt((log2Up(passes) max 1) bits)) init(0)
  val passFirstR = RegInit(True) // reset = pass-0 levels (stable pre-START)
  val passLastR = RegInit(if (passes == 1) True else False)
  val busyR = RegInit(False)
  val doneR = RegInit(False)
  // Sticky drain-done: the writer's 1-cycle done may precede
  // passDone (fast memory vs the engine's seed-pad drain tail), so latch it
  // wherever it arrives; the fence consumes (and clears) the flag.
  val writerDoneSeen = RegInit(False)
  when(io.writerDone) {
    writerDoneSeen := True
  }
  // Sticky prelude servants (hold until the handshake, never a pulse
  // the other side can miss).
  // Seed-chunk cursor: the seed read is spillBeats single-beat chunks.
  // beatIdx counts ACCEPTED chunks; readerFired (kept for the formal pin)
  // means all of them were accepted. The servant stays valid across the
  // prelude AND WaitPass: early chunks may still be unconsumed (engine idle
  // or mid-pass) when the prelude's exit conditions are otherwise met, and
  // the reader's accept gate (flushable gearbox empty = received AND
  // consumed) paces the rest at exactly the engine's consumption rate.
  val beatBits = log2Up(spillBeats + 1) max 1
  val beatIdx = Reg(UInt(beatBits bits)) init(0)
  val readerFired = RegInit(False)
  val writerFired = RegInit(False)
  val fetchSeen = RegInit(False)
  val biasPulsed = RegInit(False)

  val isLast = cnt === U(passes - 1, cnt.getWidth bits)
  val preludeEntry = (state === State.sPrelude) && (prevState =/= State.sPrelude)

  when(preludeEntry) {
    beatIdx := 0
    readerFired := False
    writerFired := False
    fetchSeen := False
    biasPulsed := False
    passFirstR := (cnt === 0)
    passLastR := isLast
  }
  val seedActive = (state === State.sPrelude) || (state === State.sWaitPass)
  io.readerCmd.valid := seedActive && (cnt =/= 0) && (beatIdx < U(spillBeats, beatBits bits))
  io.readerCmd.address := (io.spillBase + (beatIdx * U(beatBytes)).resize(addrWidth bits))
  io.readerCmd.length := U(0, 16 bits)
  when(io.readerCmd.fire) {
    beatIdx := beatIdx + 1
    when(beatIdx === U(spillBeats - 1, beatBits bits)) {
      readerFired := True
    }
  }
  io.writerCmd.valid := (state === State.sPrelude) && !isLast && !writerFired
  io.writerCmd.address := io.spillBase
  io.writerCmd.length := U(spillBeats - 1, 16 bits)
  when(io.writerCmd.fire) {
    writerFired := True
  }
  // W-slice refetch (passes > 0), suppressed under residency (fail-safe).
  // Lesson: refetchW must assert one cycle AFTER the pass index is
  // stable — the fetch plane addresses from the passIdx register, which
  // settles the cycle after cnt advances. Firing on the entry cycle would
  // re-fetch the PREVIOUS slice (same-cycle stale address). preludeHeld is
  // True from the second Prelude cycle (register delay by construction).
  val preludeHeld = RegInit(False)
  preludeHeld := (state === State.sPrelude)
  io.refetchW := (state === State.sPrelude) && (cnt =/= 0) && !fetchSeen && !io.residentMode && preludeHeld
  when(io.wFetchFire && state === State.sPrelude && cnt =/= 0) {
    fetchSeen := True
  }
  io.biasReArm := (state === State.sPrelude) && !biasPulsed
  when(state === State.sPrelude && !biasPulsed) {
    biasPulsed := True
  }
  // Same once-per-prelude flag: restartA rides the bias pulse cycle.
  io.restartA := (state === State.sPrelude) && (cnt =/= 0) && !biasPulsed && !io.residentMode

  val readerOk = (cnt === 0) || readerFired
  val writerOk = isLast || writerFired
  val fetchOk = (cnt === 0) || fetchSeen

  io.passFirst := passFirstR
  io.passLast := passLastR
  io.passIdx := cnt
  io.busy := busyR
  io.done := doneR
  doneR := False

  switch(state) {
    is(State.sIdle) {
      when(io.start && !busyR) {
        busyR := True
        cnt := 0
        state := State.sPrelude
      }
    }
    is(State.sPrelude) {
      // Scale lesson (K64-P8 stall at pass 2): never exit on the
      // prelude ENTRY cycle. The servant flags (readerFired/writerFired/
      // fetchSeen) still hold the PREVIOUS non-zero prelude's True values
      // during that cycle (preludeEntry clears them the same cycle, taking
      // effect one cycle later), so an entry-cycle exit would skip the
      // whole prelude — no seed/drain command, no W refetch, no restartA —
      // and stall in sWaitPass on A beats that never come. P == 2 could
      // never trip it (prelude 0 leaves the cnt!=0 servants False, so
      // prelude 1 genuinely waited); P >= 3 stalls. preludeHeld is True
      // from the second prelude cycle, once the clearing has landed. (The
      // beat cursor resets to 0 at the entry, so it can never fake a
      // serviced prelude on the entry cycle either.)
      when(preludeHeld && readerOk && writerOk && fetchOk) {
        state := State.sWaitPass
      }
    }
    is(State.sWaitPass) {
      when(io.passDone) {
        when(passLastR) {
          state := State.sFinish
        } otherwise {
          state := State.sWaitFence
        }
      }
    }
    is(State.sWaitFence) {
      // Same-address RAW fence: pass p+1 reads only what pass p wrote. The
      // latched done covers a pulse that arrived before the fence opened.
      when(writerDoneSeen) {
        writerDoneSeen := False
        cnt := cnt + 1
        state := State.sPrelude
      }
    }
    is(State.sFinish) {
      doneR := True
      busyR := False
      state := State.sIdle
    }
  }
}
