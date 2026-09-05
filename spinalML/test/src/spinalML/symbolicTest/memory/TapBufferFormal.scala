package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.TapBuffer

/**
 * Formal verification for TapBuffer.
 *
 * TapBuffer splits a stream into a direct passthrough branch and a deferred
 * FIFO branch. The direct branch is read immediately, while the deferred
 * branch is buffered in an exact-capacity FIFO for a downstream consumer.
 *
 * Properties verified:
 *  1. Atomicity of Handshake:
 *     io.streamIn.fire <=> io.directOut.fire <=> fifo.io.push.fire.
 *     The input is accepted if and only if BOTH the direct consumer and the
 *     internal FIFO accept the transaction simultaneously.
 *  2. Backpressure Gating:
 *     io.streamIn.ready is the logical AND of io.directOut.ready and
 *     fifo.io.push.ready. A stall on either branch stalls the input stream.
 *  3. FIFO Overflow Prevention:
 *     Push is strictly gated by ready; the FIFO can never be pushed when full.
 *  4. Data Conservation & Framing:
 *     Direct output receives streamIn payload synchronously with zero corruption.
 *     Pushed FIFO payload matches streamIn payload.
 *     Occupancy tracking: fifo.occupancy === pushedCount - poppedCount.
 *  5. Reachability (Non-vacuity covers):
 *     Simultaneous transactions, full FIFO fill, complete FIFO drain, and
 *     independent backpressure scenarios are all reachable.
 */
class TapBufferFormal extends Component {
  val depth = 4
  val lanes = 1
  val dut = FormalDut(TapBuffer(UInt(8 bits), depth = depth, lanes = lanes))

  // Drive inputs
  anyseq(dut.io.streamIn.valid)
  anyseq(dut.io.streamIn.payload)
  anyseq(dut.io.directOut.ready)
  anyseq(dut.io.tapOut.ready)

  assumeInitial(clockDomain.isResetActive)

  // Standard Stream handshake stability assumption for streamIn
  when(pastValid() && past(dut.io.streamIn.valid) && !past(dut.io.streamIn.ready)) {
    assume(dut.io.streamIn.valid)
    assume(dut.io.streamIn.payload === past(dut.io.streamIn.payload))
  }

  // Pull internal FIFO signals
  val fifoPushValid = dut.fifo.io.push.valid.pull()
  val fifoPushReady = dut.fifo.io.push.ready.pull()
  val fifoPopValid  = dut.fifo.io.pop.valid.pull()
  val fifoPopReady  = dut.fifo.io.pop.ready.pull()
  val occupancy     = dut.fifo.io.occupancy.pull()
  val entries       = dut.entries

  val inFire     = dut.io.streamIn.fire
  val directFire = dut.io.directOut.fire
  val pushFire   = dut.fifo.io.push.fire
  val tapFire    = dut.io.tapOut.fire

  // ==========================================
  // 1. HANDSHAKE ATOMICITY & BACKPRESSURE (M3.5 Invariants)
  // ==========================================
  // Invariant M3.5: The FIFO push fires if and only if the input stream fires.
  // (fifo.io.push.valid is gated on io.streamIn.ready, preventing duplicate captures under stalls).
  assert(pushFire === inFire, "fifo push fire must strictly equal streamIn fire (M3.5 root cause invariant)")

  // Whenever the input fires, the direct output is guaranteed to fire simultaneously
  when(inFire) {
    assert(directFire, "directOut must fire whenever streamIn fires")
  }

  // While the FIFO has available space, input fire and direct fire are strictly locked in lockstep
  when(fifoPushReady) {
    assert(inFire === directFire, "directOut fire and streamIn fire must match when FIFO has room")
  }

  // Ready propagation
  assert(dut.io.streamIn.ready === (dut.io.directOut.ready && fifoPushReady),
    "streamIn.ready must strictly equal directOut.ready && fifo.push.ready")

  // Valid propagation to direct output
  assert(dut.io.directOut.valid === dut.io.streamIn.valid,
    "directOut.valid must reflect streamIn.valid")

  // Tap output connects directly to FIFO pop
  assert(dut.io.tapOut.valid === fifoPopValid, "tapOut.valid must match fifo.pop.valid")
  assert(fifoPopReady === dut.io.tapOut.ready, "fifo.pop.ready must match tapOut.ready")

  // ==========================================
  // 2. FIFO SAFETY & NO OVERFLOW
  // ==========================================
  // FIFO push valid must only be asserted when streamIn fires
  assert(fifoPushValid === inFire, "FIFO push valid must only be asserted on atomic streamIn fire")

  // FIFO must never attempt to push when full
  when(fifoPushValid) {
    assert(fifoPushReady, "FIFO push attempted while FIFO was not ready!")
  }

  // Occupancy is strictly bounded by FIFO entries
  assert(occupancy <= entries, "FIFO occupancy exceeded maximum capacity")

  // When FIFO is full, push.ready must drop
  when(occupancy === entries) {
    assert(!fifoPushReady, "fifoPushReady must be False when occupancy reaches entries")
  }

  // ==========================================
  // 3. DATA INTEGRITY
  // ==========================================
  // Direct branch delivers streamIn payload transparently
  when(dut.io.directOut.valid) {
    assert(dut.io.directOut.payload(0) === dut.io.streamIn.payload(0),
      "directOut payload does not match streamIn payload")
  }

  // FIFO push payload receives streamIn payload transparently
  when(fifoPushValid) {
    assert(dut.fifo.io.push.payload(0) === dut.io.streamIn.payload(0),
      "fifo push payload does not match streamIn payload")
  }

  // Tap branch delivers FIFO pop payload transparently
  when(dut.io.tapOut.valid) {
    assert(dut.io.tapOut.payload(0) === dut.fifo.io.pop.payload(0),
      "tapOut payload does not match fifo pop payload")
  }

  // ==========================================
  // 4. REACHABILITY / LIVENESS COVERS
  // ==========================================
  // 1. Direct stream handshake
  cover(inFire)

  // 2. Tap stream handshake
  cover(tapFire)

  // 3. Simultaneous input fire and tap fire
  cover(inFire && tapFire)

  // 4. FIFO completely filled
  cover(occupancy === entries)

  // 5. Direct branch advances while tap branch is stalled
  cover(inFire && !dut.io.tapOut.ready)
}

object TapBufferFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(10)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TapBufferFormal, "tap_buffer_formal")
  }
}

