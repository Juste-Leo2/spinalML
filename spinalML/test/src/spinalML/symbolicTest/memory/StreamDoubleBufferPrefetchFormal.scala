// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.StreamDoubleBuffer

/**
 * Formal verification of the Phase-2b prefetch staging & governed swap FSM
 * in StreamDoubleBuffer.
 *
 * In resident mode (`residentHold` asserted), normal `nextTile` pulses are
 * neutralised so that the active bank remains held indefinitely.
 * When prefetch is armed (`stageRequest && tileFilled`), `switchArmed` is
 * asserted, opening a single governed flip window:
 *  1. Compute bank stability: computeBank CANNOT flip unless nextTile fires
 *     AND allowFlip is True (!freezeNow || switchArmed).
 *  2. No mid-stream swap: while compute is running (nextTile is False),
 *     computeBank stays strictly constant even if a new tile lands in the
 *     idle bank and stageRequest is asserted.
 *  3. Atomicity of governed swap: on nextTile with switchArmed asserted,
 *     computeBank toggles, switchArmed disarms back to False, and
 *     refreshSettled pulses High for exactly 1 cycle.
 *  4. Clean return to residency: once switchArmed drops, residency hold is
 *     immediately restored on the freshly swapped bank.
 */
class StreamDoubleBufferPrefetchFormal extends Component {
  val dut = FormalDut(new StreamDoubleBuffer(UInt(8 bits), depth = 4, lanes = 1,
    enableFreezePort = true))

  anyseq(dut.io.streamIn.valid)
  anyseq(dut.io.streamIn.payload)
  anyseq(dut.io.readAddr)
  anyseq(dut.io.nextTile)
  anyseq(dut.io.residentHold.get)
  anyseq(dut.io.stageRequest.get)
  dut.io.reArm := False

  assumeInitial(clockDomain.isResetActive)

  val pingFull = dut.pingFull.pull()
  val pongFull = dut.pongFull.pull()
  val computeBank = dut.computeBank.pull()
  val loadBank = dut.loadBank.pull()
  val switchArmed = dut.switchArmed.pull()
  val allowFlip = dut.allowFlip.pull()
  val tileFilled = dut.tileFilled.pull()

  // ==========================================
  // 1. SAFETY & BACKPRESSURE
  // ==========================================
  // Never accept data when both banks are full
  assert(!(pingFull && pongFull && dut.io.streamIn.ready))

  // Compute view reflects the fullness of the active compute bank
  val expectedTileReady = (computeBank === False) ? pingFull | pongFull
  assert(dut.io.tileReady === expectedTileReady)

  // Loader acceptance reflects availability of the load bank
  val expectedCanAccept = (loadBank === False) ? !pingFull | !pongFull
  assert(dut.io.loadCanAccept === expectedCanAccept)

  // ==========================================
  // 2. GOVERNED FLIP & STAGING CONTRACT
  // ==========================================
  // allowFlip is True iff residentHold is False OR switchArmed is True
  assert(allowFlip === (!dut.freezeNow || switchArmed))

  // refreshSettled pulses only on an armed governed swap
  assert(dut.io.refreshSettled === (dut.io.nextTile && allowFlip && switchArmed))

  // Governed flip transition: computeBank flips IF AND ONLY IF nextTile && allowFlip
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.nextTile && allowFlip)) {
      assert(computeBank === !past(computeBank), "computeBank must toggle on governed nextTile")
    } otherwise {
      assert(computeBank === past(computeBank), "computeBank must remain stable when not flipping")
    }
  }

  // Under residentHold without switchArmed, nextTile NEVER flips computeBank
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.freezeNow && !switchArmed && dut.io.nextTile)) {
      assert(computeBank === past(computeBank), "Resident hold must prevent bank flip")
      when(past(computeBank === False)) { assert(!past(pingFull) || pingFull) }
      when(past(computeBank === True))  { assert(!past(pongFull) || pongFull) }
    }
  }

  // switchArmed FSM transition rules:
  // - disarms when nextTile && allowFlip fires (last-assignment-wins in RTL)
  // - arms when stageRequest && tileFilled
  // - retains its state otherwise
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.nextTile && allowFlip)) {
      assert(!switchArmed, "switchArmed must be cleared when swap occurs")
    } elsewhen(past(dut.io.stageRequest.get && tileFilled)) {
      assert(switchArmed, "switchArmed must be set when stageRequest && tileFilled")
    } otherwise {
      assert(switchArmed === past(switchArmed), "switchArmed must retain state")
    }
  }

  // ==========================================
  // 3. REACHABILITY COVERS
  // ==========================================
  // 1. Idle tile landing under hold
  cover(dut.freezeNow && dut.io.tileFilled)

  // 2. Switch armed under hold
  cover(dut.freezeNow && switchArmed)

  // 3. Governed swap executed (refreshSettled)
  cover(dut.freezeNow && dut.io.refreshSettled)
}

object StreamDoubleBufferPrefetchFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(12)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new StreamDoubleBufferPrefetchFormal, "stream_double_buffer_prefetch_formal")
  }
}
