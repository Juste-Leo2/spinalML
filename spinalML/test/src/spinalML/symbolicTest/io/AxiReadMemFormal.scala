// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.io

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.io.AxiReadMem

/**
 * Formal verification for AxiReadMem:
 *  - Serves AXI4 read requests from an internal memory.
 *  - Maps virtual addresses to physical BRAM indices with defensive clamping.
 *  - Handles BRAM write port from UART bridge.
 *
 * Properties verified:
 *  1. Ready Guard:
 *     - ar.ready <=> !r.valid (cannot accept new read request while a burst is active).
 *  2. Burst Beat Counting & RLAST Invariant:
 *     - Exactly ar.len + 1 beats are delivered on the R channel.
 *     - r.last is asserted IF AND ONLY IF it is the final beat of the burst.
 *  3. Read Channel Handshake & Stall Stability:
 *     - When r.valid is high and r.ready is low (master stalls),
 *       r.valid, r.id, r.last, r.resp remain strictly constant.
 *  4. Reachability / Covers:
 *     - Complete single-beat transfer (len = 0).
 *     - Complete multi-beat burst (len > 0).
 *     - Master stall during burst.
 *     - Write transaction via wrEnable.
 */
class AxiReadMemFormal extends Component {
  val memoryWords = 8
  val axiConfig = Axi4Config(
    addressWidth = 32,
    dataWidth    = 32,
    idWidth      = 2
  )

  val dut = FormalDut(new AxiReadMem(
    axiConfig   = axiConfig,
    memoryWords = memoryWords,
    imgBase     = 0x1000,
    weightBase  = 0x2000
  ))

  // Drive inputs
  anyseq(dut.io.axi.ar.valid)
  anyseq(dut.io.axi.ar.payload)
  anyseq(dut.io.axi.r.ready)
  anyseq(dut.io.wrEnable)
  anyseq(dut.io.wrAddr)
  anyseq(dut.io.wrData)

  assumeInitial(clockDomain.isResetActive)

  // ==========================================
  // ENVIRONMENT ASSUMPTIONS
  // ==========================================
  // AR channel stability when stalled
  when(pastValid() && past(dut.io.axi.ar.valid) && !past(dut.io.axi.ar.ready)) {
    assume(dut.io.axi.ar.valid)
    assume(dut.io.axi.ar.payload === past(dut.io.axi.ar.payload))
  }

  // Bound burst length to fit comfortably within BMC depth (max 3 beats = len <= 2)
  assume(dut.io.axi.ar.payload.len <= 2)

  // No concurrent BRAM writes while read transactions are active
  when(dut.io.axi.r.valid || dut.io.axi.ar.valid) {
    assume(!dut.io.wrEnable)
  }

  // ==========================================
  // 1. READY GUARD
  // ==========================================
  assert(dut.io.axi.ar.ready === !dut.io.axi.r.valid,
    "ar.ready must be strictly !r.valid")

  // ==========================================
  // 2. STALL STABILITY (AXI4 Specification)
  // ==========================================
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(dut.io.axi.r.valid) && !past(dut.io.axi.r.ready)) {
      assert(dut.io.axi.r.valid, "r.valid dropped while stalled by master")
      assert(dut.io.axi.r.payload.id === past(dut.io.axi.r.payload.id), "r.id changed during stall")
      assert(dut.io.axi.r.payload.last === past(dut.io.axi.r.payload.last), "r.last changed during stall")
      assert(dut.io.axi.r.payload.resp === past(dut.io.axi.r.payload.resp), "r.resp changed during stall")
      when(!past(dut.io.wrEnable)) {
        assert(dut.io.axi.r.payload.data === past(dut.io.axi.r.payload.data), "r.data changed during stall")
      }
    }
  }

  // ==========================================
  // 3. BURST BEAT COUNTING & RLAST INTEGRITY
  // ==========================================
  val busy         = RegInit(False)
  val expectedLen  = Reg(UInt(8 bits)) init 0
  val beatsDeliv   = Reg(UInt(8 bits)) init 0

  when(dut.io.axi.ar.fire) {
    busy        := True
    expectedLen := dut.io.axi.ar.payload.len
    beatsDeliv  := 0
  }

  when(busy && dut.io.axi.r.fire) {
    beatsDeliv := beatsDeliv + 1
    when(dut.io.axi.r.payload.last) {
      assert(beatsDeliv === expectedLen, "r.last asserted on incorrect beat count")
      busy := False
    } otherwise {
      assert(beatsDeliv < expectedLen, "burst exceeded expected beats without r.last")
    }
  }

  when(dut.io.axi.r.valid) {
    assert(busy, "r.valid asserted without an active burst")
  }

  // ==========================================
  // 4. COVER PROPERTIES (REACHABILITY)
  // ==========================================
  // Single beat read
  cover(dut.io.axi.ar.fire && (dut.io.axi.ar.payload.len === 0))
  // Multi beat read completion
  cover(dut.io.axi.r.fire && dut.io.axi.r.payload.last && (expectedLen > 0))
  // Master stall during read
  cover(dut.io.axi.r.valid && !dut.io.axi.r.ready)
  // BRAM write
  cover(dut.io.wrEnable)
}

object AxiReadMemFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(180)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AxiReadMemFormal, "axi_read_mem_formal")
  }
}
