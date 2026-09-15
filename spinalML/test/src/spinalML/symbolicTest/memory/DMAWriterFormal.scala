// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.memory.{DMAWriter, WriteRequest}

/**
 * Formal model of the burst-splitting DMAWriter.
 *
 * The writer chains INCR bursts of at most 256 beats, clipped at 4 KiB page
 * boundaries, strictly serialized against write data draining.
 * Properties verified here:
 *
 *  1. Structural: cmd/aw/w/b handshake relations against internal counters.
 *  2. Burst legality: aw.len matches burstLen-1, never exceeds the remaining
 *     beat count nor the distance to the next 4 KiB boundary.
 *  3. Contiguity: chained bursts of one command continue at the address that
 *     follows the previous burst (INCR semantics preserved across splits).
 *  4. Last Beat Integrity: w.last is asserted IF AND ONLY IF burstRemain === 1.
 *  5. Response Accounting: done pulse occurs only after all B responses are collected.
 */
class DMAWriterFormal extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 0)
  // maxBurstBeats = 4 shrinks the burst-splitting loop for fast BMC
  val dut = FormalDut(new DMAWriter(UInt(8 bits), shape = Seq(4), inLanes = 4, axiConfig, maxBurstBeats = 4))

  anyseq(dut.io.cmd.valid)
  anyseq(dut.io.cmd.payload)
  anyseq(dut.io.inStream.stream.valid)
  anyseq(dut.io.inStream.stream.payload)
  anyseq(dut.io.axiMaster.aw.ready)
  anyseq(dut.io.axiMaster.w.ready)
  anyseq(dut.io.axiMaster.b.valid)
  anyseq(dut.io.axiMaster.b.payload)

  // Internal state handles
  val remaining   = dut.remaining.pull()
  val burstRemain = dut.burstRemain.pull()
  val burstLen    = dut.burstLen.pull()
  val beatsToBnd  = dut.beatsToBoundary.pull()
  val addrRegH    = dut.addrReg.pull()
  val pendingB    = dut.pendingB.pull()

  assumeInitial(clockDomain.isResetActive)

  // ==========================================
  // ENVIRONMENT ASSUMPTIONS
  // ==========================================
  // AXI4 requires INCR burst start addresses to be beat-aligned
  assume(dut.io.cmd.payload.address(0, 2 bits) === 0)

  // B responses only occur when there are pending bursts in flight
  when(dut.io.axiMaster.b.valid) {
    assume(pendingB =/= 0)
  }

  // Stable input handshakes
  when(pastValid() && past(dut.io.cmd.valid) && !past(dut.io.cmd.ready)) {
    assume(dut.io.cmd.valid)
    assume(dut.io.cmd.payload === past(dut.io.cmd.payload))
  }
  when(pastValid() && past(dut.io.inStream.stream.valid) && !past(dut.io.inStream.stream.ready)) {
    assume(dut.io.inStream.stream.valid)
    assume(dut.io.inStream.stream.payload === past(dut.io.inStream.stream.payload))
  }

  // ==========================================
  // 1. STRUCTURAL PROPERTIES
  // ==========================================
  when(pastValid()) {
    assert(dut.io.cmd.ready === ((remaining === 0) && (burstRemain === 0) && (pendingB === 0)))
    assert(dut.io.axiMaster.aw.valid === ((remaining =/= 0) && (burstRemain === 0)))

    when(burstRemain =/= 0) {
      assert(dut.io.axiMaster.w.valid === dut.io.inStream.stream.valid)
      assert(dut.io.inStream.stream.ready === dut.io.axiMaster.w.ready)
    } otherwise {
      assert(!dut.io.axiMaster.w.valid)
      assert(!dut.io.inStream.stream.ready)
    }
  }

  // ==========================================
  // 2. BURST LEGALITY
  // ==========================================
  when(dut.io.axiMaster.aw.valid) {
    assert(dut.io.axiMaster.aw.addr === addrRegH)
    assert(dut.io.axiMaster.aw.len === (burstLen - 1).resize(8 bits))
    assert(burstLen <= remaining)
    assert(burstLen >= 1)
    // No burst may cross a 4 KiB boundary (AXI4 protocol rule)
    assert((dut.io.axiMaster.aw.len.expand + 1) <= beatsToBnd)
  }

  // ==========================================
  // 3. W.LAST ASSERTION
  // ==========================================
  when(dut.io.axiMaster.w.valid) {
    assert(dut.io.axiMaster.w.last === (burstRemain === 1))
  }

  // ==========================================
  // 4. CHAINED-BURST CONTIGUITY
  // ==========================================
  val awIdx = Reg(UInt(8 bits)) init (0)
  when(dut.io.cmd.fire) {
    awIdx := 0
  } elsewhen (dut.io.axiMaster.aw.fire) {
    awIdx := awIdx + 1
  }
  when(pastValid() && past(dut.io.axiMaster.aw.fire) && dut.io.axiMaster.aw.fire && past(awIdx =/= 0)) {
    val prevAddr = past(dut.io.axiMaster.aw.addr)
    val prevBeats = past(dut.io.axiMaster.aw.len).expand + 1
    assert(dut.io.axiMaster.aw.addr === (prevAddr + prevBeats * 4).resized)
  }

  // ==========================================
  // 5. RESPONSE ACCOUNTING
  // ==========================================
  // Independent B counter: pendingB must always equal (accepted AWs - received
  // Bs) for the current command. This catches the simultaneous AW/B collision
  // class of bug that the response assumption above would otherwise mask (a
  // lost increment makes pendingB underflow on the next response).
  val bCnt = Reg(UInt(8 bits)) init (0)
  when(dut.io.cmd.fire) {
    bCnt := 0
  } elsewhen (dut.io.axiMaster.b.fire) {
    bCnt := bCnt + 1
  }

  when(pastValid()) {
    assert(pendingB.resize(9 bits) + bCnt.resize(9 bits) === awIdx.resize(9 bits))
  }

  cover(dut.io.axiMaster.aw.fire && dut.io.axiMaster.b.fire)

  when(dut.io.done) {
    assert(remaining === 0)
    assert(burstRemain === 0)
    assert(pendingB === 0)
  }
}

object DMAWriterFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(180)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAWriterFormal, "dma_writer_formal")
  }
}
