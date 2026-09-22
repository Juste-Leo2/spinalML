// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.nn

import spinal.core._
import spinal.core.formal._
import spinalML.nn.SpillPassController

/**
 * S2d formal pinning of the S2b spill pass controller
 * (docs/ddr_final_impl.md §S2, docs/ddr_impl.md §5.3).
 *
 * Isolated safety proof: no AXI, no memory, no Sequential — every input is
 * symbolic (`anyseq`), so the properties hold under EVERY legal environment,
 * not just the cooperative one. `passes = 3` is the smallest configuration
 * that exercises all three pass roles: pass 0 (no seed, no W refetch), the
 * middle pass (seed + drain + refetch + restartA + both commands), and the
 * final pass (seed, no drain). `addrWidth = 8` / `spillBeats = 1` keep the
 * command datapath and the state space tiny for fast BMC.
 *
 * Properties (all `pastValid()`-guarded):
 *  1. Command contract: spillBeats single-beat seed chunks per pass p>0
 *     (one command for the spillBeats=1 configuration proven here), one
 *     drain command per pass p<P-1, both at `spillBase` (+ chunk stride),
 *     never in pass 0 / final pass.
 *  2. Prelude service completeness AND the S2d entry-cycle fix: leaving the
 *     prelude requires `past(preludeHeld)` (never on the entry cycle, where
 *     the servant flags still hold the previous prelude's values) and the
 *     service evidence of the pass being closed (reader/writer/fetch fires,
 *     bias pulse, restartA pulse for p>0, none for p=0).
 *  3. `refetchW` sticky discipline (NN-01): only p>0, suppressed under
 *     residency, held until `wFetchFire`.
 *  4. One-shot pulses: `biasReArm`/`restartA` pulse at most once per prelude;
 *     command fires at most once per prelude.
 *  5. FSM flow: idle leaves only on START, waitPass only on passDone,
 *     waitFence only on writerDone (current pulse or the S2e sticky latch
 *     for a pre-fence pulse); `cnt` advances by one on the fence only,
 *     stays bounded; `passFirst`/`passLast` latch the pass role; `done` is a
 *     single pulse per run with `busy` cleared.
 *  6. Residency fail-safe: no `refetchW`/`restartA` under `residentMode`
 *     (the documented STREAM_PER_PASS contract: stall loudly, never corrupt).
 */
class SpillPassControllerFormal extends Component {
  val dut = FormalDut(SpillPassController(passes = 3, addrWidth = 8, spillBeats = 1))

  // ---- Symbolic environment -------------------------------------------
  anyseq(dut.io.start)
  anyseq(dut.io.spillBase)
  anyseq(dut.io.passDone)
  anyseq(dut.io.writerDone)
  anyseq(dut.io.wFetchFire)
  anyseq(dut.io.residentMode)
  anyseq(dut.io.readerCmd.ready)
  anyseq(dut.io.writerCmd.ready)

  assumeInitial(clockDomain.isResetActive)

  // START is a command-boundary pulse (rising edge), never held across the
  // busy span. A held START is not the host contract (CSR write pulse).
  when(pastValid() && past(dut.io.start)) {
    assume(!dut.io.start)
  }

  // ---- State handles ---------------------------------------------------
  val state        = dut.state.pull().asBits.asUInt
  val cnt          = dut.cnt.pull()
  val preludeEntry = dut.preludeEntry.pull()
  val preludeHeld  = dut.preludeHeld.pull()
  val passFirstR   = dut.passFirstR.pull()
  val passLastR    = dut.passLastR.pull()
  val readerFired  = dut.readerFired.pull()
  val writerFired  = dut.writerFired.pull()
  val fetchSeen    = dut.fetchSeen.pull()
  val biasPulsed   = dut.biasPulsed.pull()
  val isLast       = dut.isLast.pull()
  val writerDoneSeen = dut.writerDoneSeen.pull()

  val stW = state.getWidth bits
  val sIdle      = U(dut.State.sIdle.position, stW)
  val sPrelude   = U(dut.State.sPrelude.position, stW)
  val sWaitPass  = U(dut.State.sWaitPass.position, stW)
  val sWaitFence = U(dut.State.sWaitFence.position, stW)
  val sFinish    = U(dut.State.sFinish.position, stW)

  def inPrelude = state === sPrelude
  val notResident = !dut.io.residentMode

  // restartA pulse capture for the current prelude (set wins over the entry
  // clear, mirroring the DUT's own flag timing: pass 1 pulses on the entry
  // cycle, later passes one cycle after).
  val restartSeen = RegInit(False)
  when(preludeEntry) { restartSeen := False }
  when(dut.io.restartA) { restartSeen := True }

  // Runtime contract (docs/ddr_impl.md §5.3): spill runs under
  // STREAM_PER_PASS. The host never toggles residency mid-run — an isolated
  // one-cycle residency glitch could otherwise suppress `restartA` while
  // `biasReArm` still pulses (they share the pulse cycle), which is outside
  // the documented usage. Residency may be high for a WHOLE run (the
  // fail-safe then stalls the prelude, as the suppression properties show).
  when(pastValid() && dut.io.busy) {
    assume(dut.io.residentMode === past(dut.io.residentMode))
  }
  // `wFetchFire` is the acceptance of the HELD refetch request (Sequential
  // wires it to `reqW.fire` on the spillRefetchW path): it cannot arrive
  // while the request is low during a p>0 prelude.
  when(dut.io.wFetchFire && inPrelude && cnt =/= 0) {
    assume(dut.io.refetchW)
  }

  // ---- 1. Command contract --------------------------------------------
  when(dut.io.readerCmd.valid) {
    assert(inPrelude && cnt =/= 0 && !readerFired, "seed command outside a p>0 prelude")
    assert(dut.io.readerCmd.address === dut.io.spillBase, "seed command address must be spillBase")
    assert(dut.io.readerCmd.length === U(0, 16 bits), "spillBeats=1 must command a single beat")
  }
  when(dut.io.writerCmd.valid) {
    assert(inPrelude && !isLast && !writerFired, "drain command outside a non-final prelude")
    assert(dut.io.writerCmd.address === dut.io.spillBase, "drain command address must be spillBase")
    assert(dut.io.writerCmd.length === U(0, 16 bits), "spillBeats=1 must command a single beat")
  }

  // ---- 2. Prelude exit: timing fix + service completeness --------------
  when(pastValid() && past(inPrelude) && !inPrelude) {
    // S2d fix pin: never exit on the prelude entry cycle (stale servants).
    assert(past(preludeHeld), "prelude exited on its entry cycle (stale servant flags)")
    when(past(cnt) =/= 0) {
      assert(past(readerFired) || past(dut.io.readerCmd.fire), "pass p>0 exited without its seed command")
      assert(past(fetchSeen) || past(dut.io.wFetchFire), "pass p>0 exited without the W-slice refetch")
    } otherwise {
      assert(!past(readerFired), "pass 0 must not seed")
      assert(!past(fetchSeen), "pass 0 must not refetch the W slice")
    }
    when(past(cnt) =/= 2) {
      assert(past(writerFired) || past(dut.io.writerCmd.fire), "non-final pass exited without its drain command")
    }
    assert(past(biasPulsed) || past(dut.io.biasReArm), "prelude exited without the bias pulse")
    when(past(cnt) =/= 0 && past(notResident)) {
      assert(restartSeen, "pass p>0 exited without the A restart pulse")
    }
  }

  // ---- 3. refetchW contract -------------------------------------------
  when(dut.io.refetchW) {
    assert(inPrelude && cnt =/= 0, "refetchW outside a p>0 prelude")
    assert(notResident, "refetchW under residency violates STREAM_PER_PASS")
  }
  // NN-01 sticky discipline: the request may drop ONLY because the fetch was
  // accepted or because residency took over (the documented fail-safe). A
  // missing/delayed `wFetchFire` alone must never release it.
  when(pastValid() && past(dut.io.refetchW) && !dut.io.refetchW) {
    assert(past(dut.io.wFetchFire) || dut.io.residentMode,
      "refetchW dropped without an accepted fetch")
  }

  // ---- 4. One-shot pulses and fires -----------------------------------
  when(pastValid() && past(dut.io.biasReArm)) { assert(!dut.io.biasReArm, "biasReArm pulsed twice") }
  when(pastValid() && past(dut.io.restartA)) { assert(!dut.io.restartA, "restartA pulsed twice") }
  when(pastValid() && past(dut.io.readerCmd.fire)) { assert(!dut.io.readerCmd.valid, "seed commanded twice") }
  when(pastValid() && past(dut.io.writerCmd.fire)) { assert(!dut.io.writerCmd.valid, "drain commanded twice") }
  when(dut.io.restartA) { assert(inPrelude && cnt =/= 0, "restartA outside a p>0 prelude") }
  when(dut.io.biasReArm) { assert(inPrelude, "biasReArm outside a prelude") }
  when(dut.io.residentMode) { assert(!dut.io.refetchW && !dut.io.restartA, "restartA under residency") }

  // ---- 5. FSM flow -----------------------------------------------------
  assert(dut.io.passIdx === cnt, "passIdx must mirror cnt")

  when(pastValid() && past(state === sIdle) && past(dut.io.start) && !past(dut.io.busy)) {
    assert(state === sPrelude && cnt === 0 && dut.io.busy, "START must enter pass-0 prelude busy")
  }
  when(pastValid() && past(state === sWaitPass) && state =/= sWaitPass) {
    assert(past(dut.io.passDone), "waitPass left without passDone")
  }
  when(pastValid() && past(state === sWaitFence) && state =/= sWaitFence) {
    assert(past(dut.io.writerDone) || past(writerDoneSeen),
      "waitFence left without writerDone (current pulse or latched early done)")
  }
  // cnt only advances on the fence, by exactly one, and never past passes-1.
  when(pastValid() && past(state =/= sIdle)) {
    when(past(state === sWaitFence) && past(dut.io.writerDone)) {
      assert(cnt === past(cnt) + 1, "cnt must advance on the fence")
      assert(past(cnt) < U(2, cnt.getWidth bits), "cnt advanced past passes-1")
    } otherwise {
      assert(cnt === past(cnt), "cnt advanced outside the fence")
    }
  }
  // Pass-role levels latch at the prelude entry, stable afterwards.
  when(pastValid() && past(preludeEntry)) {
    assert(passFirstR === (cnt === 0), "passFirst must latch the pass-0 role")
    assert(passLastR === isLast, "passLast must latch the final-pass role")
  }
  // Single completion pulse per run, busy released with it.
  when(pastValid() && past(state === sFinish)) { assert(dut.io.done && !dut.io.busy, "no done on finish") }
  when(dut.io.done) { assert(state === sIdle, "done must return the FSM to idle") }

  // ---- 6. Reachability (non-vacuity) ----------------------------------
  cover(state === sIdle)
  cover(inPrelude && cnt === 0)
  cover(inPrelude && cnt === 1 && !passFirstR && !passLastR)
  cover(inPrelude && cnt === 2 && passLastR)
  cover(state === sWaitPass)
  cover(state === sWaitFence)
  cover(state === sFinish)
  cover(dut.io.readerCmd.fire)
  cover(dut.io.writerCmd.fire)
  cover(dut.io.readerCmd.fire && dut.io.writerCmd.fire)
  cover(dut.io.refetchW)
  cover(dut.io.restartA)
  cover(dut.io.biasReArm)
  cover(dut.io.restartA && dut.io.biasReArm)
  cover(dut.io.done)
  cover(dut.io.residentMode)
}

object SpillPassControllerFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(40)
      .withTimeout(300)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.Boolector)))
      .workspacePath("formal")
      .doVerify(new SpillPassControllerFormal, "spill_pass_controller_formal")
  }
}
