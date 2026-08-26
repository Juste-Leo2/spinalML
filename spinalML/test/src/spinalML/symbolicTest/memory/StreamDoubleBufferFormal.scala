package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.StreamDoubleBuffer

class StreamDoubleBufferFormal extends Component {
  val dut = new StreamDoubleBuffer(UInt(8 bits), depth = 4, lanes = 1)
  
  // Inject random inputs
  anyseq(dut.io.streamIn.valid)
  anyseq(dut.io.streamIn.payload)
  anyseq(dut.io.readAddr)
  anyseq(dut.io.nextTile)
  // One-shot contract: no command boundary inside the formal window.
  dut.io.reArm := False
  
  // Pull internal signals across hierarchy to avoid violations
  val pingFull = dut.pingFull.pull()
  val pongFull = dut.pongFull.pull()
  val computeBank = dut.computeBank.pull()
  val loadCounterValue = dut.loadCounter.value.pull()
  
  // ==========================================
  // SAFETY PROPERTIES (No data loss or corruption)
  // ==========================================
  
  // 1. Backpressure Check: If both banks are full, the stream must NOT be ready to accept data.
  // This prevents overwriting data that hasn't been consumed.
  assert(!(pingFull && pongFull && dut.io.streamIn.ready))
  
  // 2. Compute View Check: The compute interface should only see tileReady if the computeBank is actually full.
  val expectedTileReady = (computeBank === False) ? pingFull | pongFull
  assert(dut.io.tileReady === expectedTileReady)
  
  // 3. (Removed tautological bounds check, Counter(4) is natively 2-bits and cannot exceed 3).
  
  // ==========================================
  // REACHABILITY (Liveness / No deadlock)
  // ==========================================
  
  // 1. Can we reach a state where both banks are full? (Meaning the stream can run ahead of compute)
  cover(pingFull && pongFull)
  
  // 2. Can we successfully consume a tile?
  cover(dut.io.nextTile && dut.io.tileReady)
}

object StreamDoubleBufferFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15) // We check up to 15 cycles since it takes 4 cycles to fill one bank
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new StreamDoubleBufferFormal, "stream_double_buffer_formal")
  }
}

/**
 * Phase-2a weight-residency proof: the optional `residentHold` input must
 * NEUTRALISE nextTile — while held, consuming a tile may neither clear the
 * consumed bank's full flag nor flip the compute pointer, so the tile stays
 * visible forever (zero-DDR re-diffusion contract). Everything unrelated to
 * nextTile (backpressure, load-side full-set, reArm-free window) keeps its
 * legacy behaviour.
 */
class StreamDoubleBufferHoldFormal extends Component {
  val dut = new StreamDoubleBuffer(UInt(8 bits), depth = 4, lanes = 1,
    enableFreezePort = true)

  anyseq(dut.io.streamIn.valid)
  anyseq(dut.io.streamIn.payload)
  anyseq(dut.io.readAddr)
  anyseq(dut.io.nextTile)
  anyseq(dut.io.residentHold.get)
  dut.io.reArm := False

  // House-playbook guard (cf. DMAReaderFormal): pin the async reset to be
  // asserted at t=0 so no mid-run free-reset window can desynchronise
  // $past history from the DUT's state transition.
  assumeInitial(clockDomain.isResetActive)

  val pingFull = dut.pingFull.pull()
  val pongFull = dut.pongFull.pull()
  val computeBank = dut.computeBank.pull()

  // Legacy invariants remain valid REGARDLESS of hold mode
  assert(!(pingFull && pongFull && dut.io.streamIn.ready))
  val expectedTileReady = (computeBank === False) ? pingFull | pongFull
  assert(dut.io.tileReady === expectedTileReady)

  // The event whose effect must vanish under hold
  val heldAdvance = dut.io.nextTile && dut.io.residentHold.get

  // Clocked views: backend `past()` only, hard-gated behind a lived-edge
  // counter: no history read until >= 4 clocked edges have been survived,
  // isolating the well-known undefined-$past warm-up window entirely.
  val livedEdges = RegInit(U(0, 2 bits))
  when(livedEdges =/= 3) { livedEdges := livedEdges + 1 }
  val warmedUp = livedEdges === 3
  val heldPrev = warmedUp && past(heldAdvance)
  val heldPrev2 = warmedUp && past(heldAdvance, 2)

  // If the edge just crossed ran under hold, the compute bank cannot move.
  assert(!heldPrev || computeBank === past(computeBank))

  // ... and the consumed bank's flag must not have been erased by nextTile
  // across that governed edge (setting via concurrent loadDone stays legal;
  // the OTHER bank carries no guarantee — legitimately empty/filling).
  when(heldPrev && past(computeBank === False)) {
    assert(!past(pingFull) || pingFull)
  }
  when(heldPrev && past(computeBank === True)) {
    assert(!past(pongFull) || pongFull)
  }

  // Reachability: two consecutive held advances keep the tile consumable
  // (also proves the warm-up cannot make the whole spec vacuous).
  cover(heldPrev2 && dut.io.tileReady)
}

object StreamDoubleBufferHoldFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new StreamDoubleBufferHoldFormal, "stream_double_buffer_hold_formal")
  }
}
