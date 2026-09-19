// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.nn

import spinal.core._
import spinal.core.formal._
import spinalML.memory.MemoryKind
import spinalML.nn.{CsrMap, MemorySpec}

/**
 * Formal pinning of the Phase-1/2 DDR plumbing contracts
 * (docs/ddr_impl.md).
 *
 * `CsrMap` and `MemorySpec` are pure-Scala elaboration-time structures, so
 * there is no temporal behavior to prove here. This spec still belongs in the
 * proof flow rather than in ScalaTest only because it machine-checks, on
 * every formal run, the low-level CSR invariants the SoC control plane
 * depends on:
 *
 *  1. Bus range: every wired address fits the 8-bit AXI-Lite control bus.
 *  2. Alignment: every wired and reserved address is 4-byte aligned
 *     (AxiLite4SlaveFactory forbids unaligned addresses).
 *  3. Uniqueness: no two wired addresses alias each other.
 *  4. Reservation: the spill/stride/status plane is disjoint from wired.
 *  5. Legacy compat: `MemorySpec` defaults still equal the historical
 *     adapter bases (BramAdapter/UartSoC `imgBase = 0x10000`,
 *     `weightBase = 0x20000`), so the default descriptor elaborates exactly
 *     like the pre-Phase-2 RTL.
 *
 * Properties 1-2 are proven universally over an anyseq selector covering all
 * twelve wired addresses; 3-5 are solver-discharged constant equalities plus
 * elaboration guards. Cover points prove the selector model is not
 * over-constrained (no vacuous proof).
 *
 * Structural note: this spec has no DUT, hence no `FormalDut()` marker — and
 * without it Spinal keeps the default ASYNC reset, whose `or posedge reset`
 * check process SymbiYosys prep rejects (async2sync TRG_WIDTH error). Like
 * `LineBuffer2DFormal`, the proof therefore lives in an explicit reset-less
 * `formalCd`; the proof asserts on registers, not combinational nets.
 */
class MemorySpecFormal extends Component {
  val clk = in Bool()
  val rst = in Bool()

  val formalCd = ClockDomain(clock = clk)

  val wired = List(
    CsrMap.Start, CsrMap.Status, CsrMap.ImgBase, CsrMap.WeightBase,
    CsrMap.Mode, CsrMap.Reload, CsrMap.TileCnt, CsrMap.Run,
    CsrMap.OutAddr, CsrMap.OutCtrl, CsrMap.DmaStatus, CsrMap.DequantScale
  )
  val reserved = List(CsrMap.SpillBase, CsrMap.OutStride, CsrMap.MemStatus)

  // Elaboration guards: pure-Scala contracts, checked on every proof run.
  require(wired.size == 12, s"CsrMap wired set changed size (got ${wired.size}, want 12)")
  require(reserved.size == 3, s"CsrMap reserved set changed size (got ${reserved.size}, want 3)")
  require((wired.toSet & reserved.toSet).isEmpty, "reserved CSR collides with a wired address")
  require(MemorySpec.default.kind == MemoryKind.OnChip, "MemorySpec default kind changed")
  require(MemorySpec.default.spillBase.isEmpty, "MemorySpec default must not declare a spill region")
  require(MemorySpec.default.capacityBytes.isEmpty, "MemorySpec default must not declare a capacity")

  new ClockingArea(formalCd) {
    // Power-on reset constraint + lived-guard (LineBuffer2DFormal pattern).
    assumeInitial(rst)
    val pastValid = Reg(Bool())
    assumeInitial(!pastValid)
    pastValid := True
    when(pastValid) {
      assume(!rst)
    }

    // Universal selector over the wired set.
    val sel = anyseq(UInt(4 bits))
    assume(sel < U(wired.size, 4 bits))
    var addr: UInt = U(wired.head, 8 bits)
    for (i <- wired.indices.tail) {
      addr = Mux(sel === U(i, 4 bits), U(wired(i), 8 bits), addr)
    }
    // Registered snapshot: asserts target registers (repo playbook rule).
    // The one-cycle delay changes nothing: the checked values are constants.
    val addrReg = RegNext(addr)

    // Reserved plane selector.
    val rSel = anyseq(UInt(2 bits))
    assume(rSel < U(reserved.size, 2 bits))
    var rAddr: UInt = U(reserved.head, 8 bits)
    for (i <- reserved.indices.tail) {
      rAddr = Mux(rSel === U(i, 2 bits), U(reserved(i), 8 bits), rAddr)
    }
    val rAddrReg = RegNext(rAddr)

    when(pastValid) {
      // 1+2. Every wired address fits the bus and is 4-byte aligned.
      assert(addrReg(1 downto 0) === 0, "wired CSR address is not 4-byte aligned")

      // 3. Pairwise uniqueness of the wired set.
      val distinct = wired.combinations(2).map {
        case Seq(a, b) => U(a, 8 bits) =/= U(b, 8 bits)
      }.reduce(_ && _)
      assert(distinct, "two wired CSR addresses alias each other")

      // 4. Reserved plane: aligned and disjoint from every wired address.
      assert(rAddrReg(1 downto 0) === 0, "reserved CSR address is not 4-byte aligned")
      val disjoint = wired.foldLeft(True)((acc, w) => acc && (rAddrReg =/= U(w, 8 bits)))
      assert(disjoint, "reserved CSR address aliases a wired address")

      // 5. Legacy-compat: default descriptor bases equal the historical ones.
      assert(U(BigInt(MemorySpec.default.imgBase), 32 bits) === U(0x10000, 32 bits),
        "MemorySpec default imgBase drifted from the historical 0x10000")
      assert(U(BigInt(MemorySpec.default.weightBase), 32 bits) === U(0x20000, 32 bits),
        "MemorySpec default weightBase drifted from the historical 0x20000")
    }

    // Non-vacuity: every wired and reserved address is selectable.
    for (i <- wired.indices) {
      cover(sel === U(i, 4 bits))
    }
    for (i <- reserved.indices) {
      cover(rSel === U(i, 2 bits))
    }
  }
}

object MemorySpecFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(120)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new MemorySpecFormal, "memory_spec_formal")
  }
}
