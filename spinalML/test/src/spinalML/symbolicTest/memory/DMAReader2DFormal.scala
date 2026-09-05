// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.memory.{DMAReader2D, FetchRequest2D}

/**
 * Formal model of the rewritten serialized-row DMAReader2D.
 *
 * The component breaks a 2D request into strictly serialized 1D row commands,
 * aligning unaligned row starts DOWN to the AXI beat and trimming the leading
 * (alignment) and trailing (overshoot) extra elements, so the element stream
 * carries exactly shape(1) elements per row. Properties verified here:
 *
 *  1. Structural / addressing: every fetched row is beat-aligned, sits one
 *     beat below the unaligned row base, and its beat budget is exactly
 *     ceil((headSkip + W) / EW); AR addresses stay beat-aligned (the design
 *     guarantees what Sequential used to assume); rows advance in order below
 *     the commanded height and completion happens only on the last row.
 *  2. Trim geometry: the latched window anchors the filter — rowSkip is the
 *     misalignment of the issued row base, rowKeepEnd = rowSkip + W - 1.
 *  3. Beat accounting (the Mnist bug class): per command, kept output beats
 *     have consecutive indices starting at rowSkip — no gap, no duplicate,
 *     order preserved — and total exactly H * (W / lanes); nothing is lost
 *     across row boundaries.
 *  4. State integrity: elemCnt stays below its row budget while a row is in
 *     flight; currentRow never exceeds the commanded height.
 *  5. Reachability (cover pass): command acceptance, output beats, full
 *     completion, unaligned rows and active trimming are reachable.
 *
 * SPEC STYLE NOTE (hard-won): only REGISTERS and top-level IO ports are ever
 * pulled from the DUT (`dut.fsm.stateReg`, `dut.rowWords`, ...) and every
 * relation is checked against signals latched at the same clock edge. The
 * counterexamples that drove this design were genuine SPEC bugs, not RTL
 * ones: a bit-slicing trap (`x(1, 2 bits)` extracts bits [2:1], NOT the low
 * two bits -- SpinalHDL syntax is (offset, width)) and an off-by-one on the
 * beat budget (-1 belongs to readerCmd.length, not to the latched rowWords).
 * Always cross-check a suspicious assertion against the generated SV before
 * suspecting the RTL. Content equivalence against a memory model is
 * deliberately out of scope: it is covered by simulation (Mnist/Mnistw4a8
 * black-box benches, 784/784 bit-exact image echo).
 */
class DMAReader2DFormal(lanes: Int = 1, alignedRowsOnly: Boolean = false) extends Component {
  val axiConfig = Axi4Config(addressWidth = 16, dataWidth = 32, idWidth = 0)
  val dut = FormalDut(new DMAReader2D(UInt(8 bits), shape = Seq(4, 4), outLanes = lanes, axiConfig))

  // Geometry constants mirrored from the DUT parameters
  val W     = 4                      // shape(1), elements per row
  val EW    = 4                      // elemsPerWord = (dataWidth/8)/(bitsWidth/8)
  val KEEPS = W / lanes              // kept output beats per row

  // FSM encodings (declaration order in DMAReader2D)
  val S_BOOT  = 0
  val S_IDLE  = 1
  val S_FETCH = 2
  val S_DRAIN = 3

  anyseq(dut.io.cmd.valid)
  anyseq(dut.io.cmd.payload)
  anyseq(dut.io.axiMaster.ar.ready)
  anyseq(dut.io.axiMaster.r.valid)
  anyseq(dut.io.axiMaster.r.payload)
  anyseq(dut.io.outStream.stream.ready)

  // ---- Pulled state: REGISTERS only (see style note) ----
  val state           = dut.fsm.stateReg.pull().asBits.asUInt
  val currentRow      = dut.currentRow.pull()
  val currentAddress  = dut.currentAddress.pull()
  val cmdHeight       = dut.cmdHeight.pull()
  val lastRow         = dut.lastRow.pull()
  val rowReqAddr      = dut.rowReqAddr.pull()
  val rowWords        = dut.rowWords.pull()
  val rowSkip         = dut.rowSkip.pull()
  val rowKeepEnd      = dut.rowKeepEnd.pull()
  val elemCnt         = dut.elemCnt.pull()
  val burstRemain     = dut.reader1D.burstRemain.pull()

  val fetching = state === S_FETCH
  val draining = state === S_DRAIN
  val outFire  = dut.io.outStream.stream.fire   // top-level ports only
  val cmdFire  = dut.io.cmd.fire
  // Widened view for comparisons: never let SpinalHDL "resize" the other
  // side DOWN to elemCnt's own (config-dependent) width.
  val eC = elemCnt.resize(12 bits)

  // ==========================================
  // ENVIRONMENT ASSUMPTIONS
  // ==========================================
  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.cmd.patchHeight > 0)

  // Multi-lane contract: whole-group trim requires group-aligned rows. With
  // 1 byte per element, every row start (base + r*stride) is beat-aligned
  // iff base and stride both are.
  if (alignedRowsOnly) {
    assume(dut.io.cmd.payload.baseAddress(0, 2 bits) === 0)
    assume(dut.io.cmd.payload.stride(0, 2 bits) === 0)
  }

  // AXI-Stream protocol: stable-valid handshakes on the control stream...
  when(pastValid() && past(dut.io.cmd.valid) && !past(dut.io.cmd.ready)) {
    assume(dut.io.cmd.valid)
    assume(dut.io.cmd.payload === past(dut.io.cmd.payload))
  }
  // ...and on R responses, which only ever belong to an outstanding burst.
  when(dut.io.axiMaster.r.valid) {
    assume(burstRemain =/= 0)
  }
  when(pastValid() && past(dut.io.axiMaster.r.valid) && !past(dut.io.axiMaster.r.ready)) {
    assume(dut.io.axiMaster.r.valid)
    assume(dut.io.axiMaster.r.payload === past(dut.io.axiMaster.r.payload))
  }

  // ==========================================
  // 1. STRUCTURAL / ADDRESSING PROPERTIES
  // ==========================================
  // While issuing a row command, rows stay in order under the height bound.
  when(pastValid() && fetching) {
    assert(currentRow < cmdHeight, "row command outside commanded height")
    assert(currentRow <= cmdHeight, "row counter exceeded height")
  }
  // At the fetch->drain edge the row command was accepted: check the latched
  // geometry against what was visible during the issuing cycle.
  when(pastValid() && past(fetching) && draining) {
    // Aligned-down address of the issued row base...
    assert(rowReqAddr(0, 2 bits) === 0, "row fetch not beat-aligned")
    assert((past(currentAddress) - rowReqAddr) < 4, "align-down skipped a beat")
    // ...and a beat budget covering head-skip + row width exactly.
    val expectedWords = ((rowSkip.resize(8 bits) +^ W +^ (EW - 1)) / EW)
    assert(rowWords === expectedWords.resized, "row beat budget mismatch")
    // Row skip is exactly the misalignment of the issued row base
    // (1 byte per element, so byte offset == element offset).
    assert(rowSkip === past(currentAddress)(0, 2 bits), "rowSkip != head skip of issued row base")
    assert(rowKeepEnd === (rowSkip +^ W - 1).resized, "keep window end mismatch")
  }
  // The design guarantees what callers previously had to assume: every AR
  // is beat-aligned (Sequential relies on this for region starts).
  when(pastValid() && dut.io.axiMaster.ar.valid) {
    assert(dut.io.axiMaster.ar.addr(0, 2 bits) === 0, "AR not beat-aligned")
  }
  // Completion happens only on the last row
  when(pastValid() && dut.io.cmd.ready) {
    assert(lastRow, "command completed before the last row")
    assert(currentRow === cmdHeight - 1, "completion row counter off")
  }

  // ==========================================
  // 2. BEAT ACCOUNTING PER COMMAND (no loss, order preserved)
  // ==========================================
  val keptThisCmd = Reg(UInt(20 bits)) init (0)
  val sawKept     = RegInit(False)
  val prevElemCnt = Reg(UInt(6 bits)) init (0)

  when(cmdFire) {
    keptThisCmd := 0
    sawKept := False
  } elsewhen (outFire) {
    keptThisCmd := keptThisCmd + 1
    when(sawKept) {
      // Consecutive kept indices: no gap, no duplicate -> order preserved
      assert(eC === (prevElemCnt + 1).resized, "kept beats not consecutive")
    } otherwise {
      // First kept beat sits exactly at the alignment skip
      assert(eC === rowSkip.resized, "first kept beat not at rowSkip")
    }
    sawKept := True
    prevElemCnt := elemCnt.resized
  }

  // A newly issued row restarts the kept-index sequence at ITS rowSkip:
  // reset the consecutiveness tracker on every fetch->drain edge (placed
  // after the counting block so a beat firing on the same edge as the new
  // row's latch is treated as that row's first kept beat).
  when(pastValid() && past(fetching) && draining) {
    sawKept := False
  }

  // Exact delivery: every row contributes exactly its kept beats
  when(pastValid() && dut.io.cmd.ready) {
    val delivered = keptThisCmd + (outFire ? U(1, 20 bits) | U(0, 20 bits))
    assert(delivered === (cmdHeight * U(KEEPS, 20 bits)).resized, "beat count mismatch over command")
  }

  // ==========================================
  // 3. STATE INTEGRITY INVARIANTS
  // ==========================================
  when(pastValid()) {
    // Meaningful once the row geometry is latched (DRAIN): during FETCH
    // rowWords still holds the PREVIOUS row's budget, so the bound below
    // would legitimately read 0 < 0 on the first rows.
    when(draining) {
      assert(eC < (rowWords * U(EW, 12 bits)), "element counter escaped its row")
    }
    assert(currentRow <= cmdHeight, "row counter exceeded height")
  }

  // ==========================================
  // 4. REACHABILITY COVERS
  // ==========================================
  val cmdCount = Reg(UInt(4 bits)) init (0)
  when(cmdFire) { cmdCount := cmdCount + 1 }

  cover(cmdFire)
  cover(outFire)
  cover(dut.io.cmd.ready)
  if (!alignedRowsOnly) cover(rowSkip =/= 0)                  // unaligned row exercised
  cover(draining && elemCnt < rowSkip)                        // trimming actively drops a beat
  cover(cmdCount >= 2)                                        // back-to-back commands
}

// Distinct top-level names: each contract gets its own formal/ workspace
class DMAReader2DFormalLanes1 extends DMAReader2DFormal(lanes = 1)
class DMAReader2DFormalLanes4Aligned extends DMAReader2DFormal(lanes = 4, alignedRowsOnly = true)

object DMAReader2DFormal {
  def main(args: Array[String]): Unit = {
    // Contract 1: lanes = 1 (production image path), free stride — full proof
    FormalConfig
      .withSymbiYosys
      .withBMC(10)
      .withTimeout(300)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAReader2DFormalLanes1, "dma_reader_2d_lanes1_bmc")

    // Reachability of the interesting events (non-vacuity)
      FormalConfig
        .withSymbiYosys
        .withCover(20)
        .withTimeout(300)
        .withDebug
        .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
        .workspacePath("formal")
        .doVerify(new DMAReader2DFormalLanes1, "dma_reader_2d_lanes1_cover")

    // Contract 2: lanes > 1 under the documented aligned-rows precondition
    FormalConfig
      .withSymbiYosys
      .withBMC(10)
      .withTimeout(300)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAReader2DFormalLanes4Aligned, "dma_reader_2d_lanes4_aligned_bmc")
  }
}
