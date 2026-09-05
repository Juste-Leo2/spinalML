// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.accelerator

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.accelerator.MLAccelerator

class MLAcceleratorFormal extends Component {
  val dut = FormalDut(MLAccelerator(axiDataWidth = 32))

  // Drive inputs symbolically
  anyseq(dut.io.axisInA.valid)
  anyseq(dut.io.axisInA.payload.data)
  anyseq(dut.io.axisInA.payload.last)

  anyseq(dut.io.axisInB.valid)
  anyseq(dut.io.axisInB.payload.data)
  anyseq(dut.io.axisInB.payload.last)

  anyseq(dut.io.axisOut.ready)

  assumeInitial(clockDomain.isResetActive)

  // Standard AMBA AXI4-Stream stability assumptions on upstream inputs
  when(pastValid() && past(dut.io.axisInA.valid) && !past(dut.io.axisInA.ready)) {
    assume(dut.io.axisInA.valid)
    assume(dut.io.axisInA.payload.data === past(dut.io.axisInA.payload.data))
    assume(dut.io.axisInA.payload.last === past(dut.io.axisInA.payload.last))
  }

  when(pastValid() && past(dut.io.axisInB.valid) && !past(dut.io.axisInB.ready)) {
    assume(dut.io.axisInB.valid)
    assume(dut.io.axisInB.payload.data === past(dut.io.axisInB.payload.data))
    assume(dut.io.axisInB.payload.last === past(dut.io.axisInB.payload.last))
  }

  // ==========================================
  // SAFETY PROPERTIES (Protocol Compliance)
  // ==========================================

  // 1. Reset cleanliness: during reset, axisOut must never assert valid
  when(clockDomain.isResetActive) {
    assert(!dut.io.axisOut.valid, "axisOut.valid asserted while in reset")
  }

  // 2. AXI4-Stream Protocol Invariant: output cannot retract valid or mutate payload while stalled
  when(pastValid() && past(dut.io.axisOut.valid) && !past(dut.io.axisOut.ready)) {
    assert(dut.io.axisOut.valid, "axisOut dropped valid without a handshake")
    assert(dut.io.axisOut.payload.data === past(dut.io.axisOut.payload.data), "axisOut mutated data under backpressure")
    assert(dut.io.axisOut.payload.last === past(dut.io.axisOut.payload.last), "axisOut mutated last under backpressure")
  }

  // ==========================================
  // REACHABILITY (Liveness / No deadlock)
  // ==========================================

  // 1. Output can fire
  cover(dut.io.axisOut.fire)

  // 2. Complete frame with TLAST can fire
  cover(dut.io.axisOut.fire && dut.io.axisOut.last)

  // 3. Handshake can occur after downstream stall
  cover(dut.io.axisOut.fire && pastValid() && !past(dut.io.axisOut.ready))
}

object MLAcceleratorFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(10)
      .withTimeout(300)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new MLAcceleratorFormal, "ml_accelerator_formal")
  }
}
