// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._
import spinal.lib._
import spinalML.memory.{FetchRequest, WriteRequest}

/**
 * S2b spill pass controller (docs/ddr_final_impl.md): owns the K-pass loop
 * of the single v1 spilling layer.
 *
 * One pass `p` on `P` slices:
 *   1. Prelude — hold `passFirst`/`passLast` for the pass; fire the seed
 *      read (p > 0) and the drain write (p < P-1) against the single M*N
 *      spill region; re-fire the W-slice fetch (p > 0, held until the fetch
 *      plane accepts — NN-01 sticky discipline); pulse `biasReArm` once
 *      (the bias stays parked from the START fetch, consumed final only).
 *   2. WaitPass — the compute drains (seed self-synchronizes by
 *      backpressure: the reader was commanded in the prelude, so a slow
 *      seed only stalls, never deadlocks).
 *   3. WaitFence (non-final) — the next pass starts only after
 *      `writerDone`: same-address RAW on the single region.
 *
 * P == 1 (full-width single pass, legal since S0) flows through with no
 * reader/writer command: seed zeros, drain straight to `io.y`.
 *
 * STREAM_PER_PASS fail-safe: `refetchW` is suppressed under residency —
 * a spill under resident/prefetch modes stalls loudly instead of silently
 * corrupting (see the S0 runtime-contract note in Sequential).
 */
case class SpillPassController(
  passes: Int,
  addrWidth: Int = 32,
  spillBeats: Int
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

    val passFirst = out Bool() // stable per pass (S1 contract)
    val passLast = out Bool() // stable per pass (S1 contract)
    val passIdx = out UInt((log2Up(passes) max 1) bits) // current pass (fetch addressing)
    val refetchW = out Bool() // re-fire the W-slice fetch (held till wFetchFire)
    val biasReArm = out Bool() // one-cycle pulse per pass prelude
    val readerCmd = master(Stream(FetchRequest(addrWidth)))
    val writerCmd = master(Stream(WriteRequest(addrWidth)))
    val busy = out Bool()
    val done = out Bool() // one-cycle pulse when the final pass drains
  }

  val state = RegInit(State.sIdle)
  val prevState = RegInit(State.sIdle)
  prevState := state
  val cnt = Reg(UInt((log2Up(passes) max 1) bits)) init(0)
  val passFirstR = RegInit(True) // reset = pass-0 levels (stable pre-START)
  val passLastR = RegInit(if (passes == 1) True else False)
  val busyR = RegInit(False)
  val doneR = RegInit(False)
  // Sticky prelude servants (NN-01: hold until the handshake, never a pulse
  // the other side can miss).
  val readerFired = RegInit(False)
  val writerFired = RegInit(False)
  val fetchSeen = RegInit(False)
  val biasPulsed = RegInit(False)

  val isLast = cnt === U(passes - 1, cnt.getWidth bits)
  val preludeEntry = (state === State.sPrelude) && (prevState =/= State.sPrelude)

  // ---- Prelude servants ------------------------------------------------
  when(preludeEntry) {
    readerFired := False
    writerFired := False
    fetchSeen := False
    biasPulsed := False
    passFirstR := (cnt === 0)
    passLastR := isLast
  }
  io.readerCmd.valid := (state === State.sPrelude) && (cnt =/= 0) && !readerFired
  io.readerCmd.address := io.spillBase
  io.readerCmd.length := U(spillBeats - 1, 16 bits)
  when(io.readerCmd.fire) {
    readerFired := True
  }
  io.writerCmd.valid := (state === State.sPrelude) && !isLast && !writerFired
  io.writerCmd.address := io.spillBase
  io.writerCmd.length := U(spillBeats - 1, 16 bits)
  when(io.writerCmd.fire) {
    writerFired := True
  }
  // W-slice refetch (passes > 0), suppressed under residency (fail-safe).
  io.refetchW := (state === State.sPrelude) && (cnt =/= 0) && !fetchSeen && !io.residentMode
  when(io.wFetchFire && state === State.sPrelude && cnt =/= 0) {
    fetchSeen := True
  }
  io.biasReArm := (state === State.sPrelude) && !biasPulsed
  when(state === State.sPrelude && !biasPulsed) {
    biasPulsed := True
  }

  val readerOk = (cnt === 0) || readerFired
  val writerOk = isLast || writerFired
  val fetchOk = (cnt === 0) || fetchSeen

  // ---- Main FSM --------------------------------------------------------
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
      when(readerOk && writerOk && fetchOk) {
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
      // Same-address RAW fence: pass p+1 reads only what pass p wrote.
      when(io.writerDone) {
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
