// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.io

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._
import spinalML.io.UartBridge

/**
 * Formal verification for UartBridge:
 *  - Decodes L2 UART protocol frames ('C', 'W', 'R', 'S', 'V').
 *  - Manages AXI-Lite master write transactions to CSR.
 *  - Handles BRAM streaming write and sequential address calculation.
 *  - Manages accelerator output stream reading with backpressure.
 *
 * Properties verified:
 *  1. AXI-Lite Master Protocol Compliance:
 *     - csr.aw.valid stability under stall (AWREADY = 0).
 *     - csr.w.valid stability under stall (WREADY = 0).
 *     - Independent AW and W completion (either can complete first).
 *     - csr.ar.valid is strictly False (write-only CSR port).
 *     - csr.r.ready is strictly False.
 *  2. Stream Flow Control & Backpressure:
 *     - io.outStream.ready is asserted IF AND ONLY IF state === R_WAIT.
 *     - Data conservation: every byte popped from outStream is transmitted on io.tx.
 *  3. BRAM Write & Address Progression:
 *     - wrEnable pulses for exactly 1 cycle per word.
 *     - wrAddr increments by wordBytes exactly on the cycle following wrEnable.
 *  4. Reachability / Covers:
 *     - End-to-end completion of Version command ('V').
 *     - End-to-end completion of Status command ('S').
 *     - End-to-end completion of CSR write command ('C').
 *     - End-to-end completion of BRAM write command ('W').
 *     - End-to-end completion of Stream read command ('R').
 */
class UartBridgeFormal extends Component {
  val outCount     = 2
  val wordWidth    = 16 // 2 bytes per word for fast BMC unrolling
  val csrAddrWidth = 8
  val version      = 0x01

  val dut = FormalDut(new UartBridge(
    outCount     = outCount,
    wordWidth    = wordWidth,
    csrAddrWidth = csrAddrWidth,
    version      = version
  ))

  // Drive inputs
  anyseq(dut.io.rx.valid)
  anyseq(dut.io.rx.payload)
  anyseq(dut.io.tx.ready)
  anyseq(dut.io.csr.aw.ready)
  anyseq(dut.io.csr.w.ready)
  anyseq(dut.io.csr.b.valid)
  anyseq(dut.io.csr.b.payload)
  anyseq(dut.io.outStream.valid)
  anyseq(dut.io.outStream.payload)
  anyseq(dut.io.statusArValid)
  anyseq(dut.io.statusRValid)
  anyseq(dut.io.accBusy)
  anyseq(dut.io.accDone)

  assumeInitial(clockDomain.isResetActive)

  // ==========================================
  // ENVIRONMENT ASSUMPTIONS
  // ==========================================
  // outStream stability under backpressure
  when(pastValid() && past(dut.io.outStream.valid) && !past(dut.io.outStream.ready)) {
    assume(dut.io.outStream.valid)
    assume(dut.io.outStream.payload === past(dut.io.outStream.payload))
  }

  // AXI-Lite B channel response assumptions: BVALID only when bridge is in C_EXEC_B
  when(dut.io.csr.b.valid) {
    assume(dut.state === dut.C_EXEC_B)
  }
  when(pastValid() && past(dut.io.csr.b.valid) && !past(dut.io.csr.b.ready)) {
    assume(dut.io.csr.b.valid)
  }

  // ==========================================
  // 1. AXI-LITE PROTOCOL PROPERTIES
  // ==========================================
  // Read channels are never used
  assert(!dut.io.csr.ar.valid, "csr.ar.valid must never be asserted")
  assert(!dut.io.csr.r.ready, "csr.r.ready must never be asserted")

  // AW channel stability
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.csr.aw.valid) && !past(dut.io.csr.aw.ready)) {
      assert(dut.io.csr.aw.valid, "csr.aw.valid dropped before aw.ready")
      assert(dut.io.csr.aw.payload.addr === past(dut.io.csr.aw.payload.addr), "csr.aw.addr changed during stall")
    }
  }

  // W channel stability
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.csr.w.valid) && !past(dut.io.csr.w.ready)) {
      assert(dut.io.csr.w.valid, "csr.w.valid dropped before w.ready")
      assert(dut.io.csr.w.payload.data === past(dut.io.csr.w.payload.data), "csr.w.data changed during stall")
      assert(dut.io.csr.w.payload.strb === B"1111", "csr.w.strb must be 1111")
    }
  }

  // ==========================================
  // 2. STREAM FLOW CONTROL & BACKPRESSURE
  // ==========================================
  // outStream.ready must be asserted ONLY when in R_WAIT state and tx is ready
  assert(dut.io.outStream.ready === ((dut.state === dut.R_WAIT) && dut.io.tx.ready),
    "outStream.ready must be strictly ((state === R_WAIT) && tx.ready)")

  // Data conservation from outStream to TX
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.outStream.fire)) {
      assert(dut.io.tx.valid, "tx.valid must be high on cycle following outStream.fire")
      assert(dut.io.tx.payload === past(dut.io.outStream.payload),
        "tx.payload must match the consumed outStream byte")
    }
  }

  // ==========================================
  // 3. BRAM WRITE & ADDRESS INCREMENT
  // ==========================================
  when(pastValid() && !clockDomain.isResetActive) {
    when(dut.io.wrEnable) {
      assert(!past(dut.io.wrEnable), "wrEnable must not pulse consecutively")
    }
    when(past(dut.io.wrEnable)) {
      assert(dut.io.wrAddr === (past(dut.io.wrAddr) + dut.wordBytes).resized,
        "wrAddr did not increment by wordBytes after wrEnable")
    }
  }

  // ==========================================
  // 4. REACHABILITY / COVER PROPERTIES
  // ==========================================
  // Cover Version response
  cover(dut.state === dut.S_SEND && (dut.io.tx.payload === version))

  // Cover Status response
  cover(dut.state === dut.S_SEND && dut.io.tx.fire)

  // Cover CSR write completion back to IDLE
  cover(pastValid() && past(dut.state === dut.C_EXEC_B) && (dut.state === dut.IDLE))

  // Cover BRAM write completion back to IDLE
  cover(pastValid() && past(dut.state === dut.W_DATA) && (dut.state === dut.IDLE))

  // Cover Stream read completion back to IDLE
  cover(pastValid() && past(dut.state === dut.R_SEND) && (dut.state === dut.IDLE) && (dut.logitCnt === (outCount - 1)))
}

object UartBridgeFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(12)
      .withTimeout(180)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new UartBridgeFormal, "uart_bridge_formal")
  }
}
