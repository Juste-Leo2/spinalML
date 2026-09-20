// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.nn

import spinal.core._
import spinal.core.formal._
import spinalML.nn.SpillPassController

/**
 * S2d bounded-liveness companion of `SpillPassControllerFormal`
 * (docs/ddr_final_impl.md §S2, docs/ddr_impl.md §5.3).
 *
 * The safety harness proves the controller can never do the wrong thing; this
 * one proves it cannot hang doing the right thing: from an accepted START,
 * with a FAIR-but-bounded environment, the run completes within a bounded
 * number of cycles and returns to idle.
 *
 * Environment model (all bounds are small and explicit):
 *  - STREAM_PER_PASS: residency is off (its stall is the documented
 *    fail-safe, not a liveness bug).
 *  - START is a one-cycle command pulse.
 *  - Every awaited handshake arrives within a small window: `passDone` while
 *    in waitPass, `writerDone` while in waitFence, `wFetchFire` while
 *    `refetchW` is held, and each spill command stream is accepted within a
 *    few cycles of asserting `valid`.
 *
 * Under those assumptions the worst-case run is ~3 passes x (prelude + wait +
 * fence) with each stage <= 4-5 cycles, so completion far below the elapsed
 * bound is reachable; the assert pins that no path exceeds it.
 */
class SpillPassControllerLivenessFormal extends Component {
  val dut = FormalDut(SpillPassController(passes = 3, addrWidth = 8, spillBeats = 1))

  anyseq(dut.io.start)
  anyseq(dut.io.spillBase)
  anyseq(dut.io.passDone)
  anyseq(dut.io.writerDone)
  anyseq(dut.io.wFetchFire)
  anyseq(dut.io.readerCmd.ready)
  anyseq(dut.io.writerCmd.ready)
  dut.io.residentMode := False

  assumeInitial(clockDomain.isResetActive)

  when(pastValid() && past(dut.io.start)) {
    assume(!dut.io.start)
  }

  // ---- State handles ---------------------------------------------------
  val state = dut.state.pull().asBits.asUInt
  val stW = state.getWidth bits
  val sIdle      = U(dut.State.sIdle.position, stW)
  val sPrelude   = U(dut.State.sPrelude.position, stW)
  val sWaitPass  = U(dut.State.sWaitPass.position, stW)
  val sWaitFence = U(dut.State.sWaitFence.position, stW)

  // ---- Bounded fairness assumptions (K = 3 cycles after first waiting) --
  val fetchWait = Reg(UInt(3 bits)) init(0)
  when(dut.io.refetchW) { fetchWait := fetchWait + 1 } otherwise { fetchWait := 0 }
  assume(dut.io.wFetchFire || !dut.io.refetchW || fetchWait < U(3, 3 bits))

  val passWait = Reg(UInt(3 bits)) init(0)
  when(state === sWaitPass) { passWait := passWait + 1 } otherwise { passWait := 0 }
  assume(dut.io.passDone || state =/= sWaitPass || passWait < U(3, 3 bits))

  val fenceWait = Reg(UInt(3 bits)) init(0)
  when(state === sWaitFence) { fenceWait := fenceWait + 1 } otherwise { fenceWait := 0 }
  assume(dut.io.writerDone || state =/= sWaitFence || fenceWait < U(3, 3 bits))

  val rWait = Reg(UInt(3 bits)) init(0)
  when(dut.io.readerCmd.valid && !dut.io.readerCmd.ready) { rWait := rWait + 1 } otherwise { rWait := 0 }
  assume(!dut.io.readerCmd.valid || dut.io.readerCmd.ready || rWait < U(3, 3 bits))

  val wWait = Reg(UInt(3 bits)) init(0)
  when(dut.io.writerCmd.valid && !dut.io.writerCmd.ready) { wWait := wWait + 1 } otherwise { wWait := 0 }
  assume(!dut.io.writerCmd.valid || dut.io.writerCmd.ready || wWait < U(3, 3 bits))

  // ---- Completion deadline --------------------------------------------
  val started  = RegInit(False)
  val doneSeen = RegInit(False)
  val elapsed  = Reg(UInt(7 bits)) init(0)
  when(dut.io.start && state === sIdle) {
    started := True
    doneSeen := False
    elapsed := 0
  } elsewhen(started && !doneSeen) {
    elapsed := elapsed + 1
  }
  when(dut.io.done) { doneSeen := True }

  when(pastValid()) {
    when(started && elapsed > U(60, 7 bits)) {
      assert(doneSeen, "spill pass controller did not complete within the bounded-fairness deadline")
    }
    when(dut.io.done) { assert(started, "done pulse without an accepted START") }
  }

  // Reachability: a full run completes under the fairness model.
  cover(doneSeen)
  cover(state === sPrelude)
  cover(state === sWaitFence)
  cover(doneSeen && state === sIdle)
}

object SpillPassControllerLivenessFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(80)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.Boolector)))
      .workspacePath("formal")
      .doVerify(new SpillPassControllerLivenessFormal, "spill_pass_controller_liveness_formal")
  }
}
