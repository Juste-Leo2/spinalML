package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.memory.{DMAReader, FetchRequest}

/**
 * Formal model of the burst-splitting DMAReader.
 *
 * The reader chains INCR bursts of at most 256 beats, clipped at 4 KiB page
 * boundaries, strictly serialized (a burst is issued only once the previous
 * one fully drained). Properties verified here:
 *
 *  1. Structural: cmd/ar handshake relations against the internal counters.
 *  2. Burst legality: ar.len matches burstLen-1, never exceeds the remaining
 *     beat count nor the distance to the next 4 KiB boundary.
 *  3. Contiguity: chained bursts of one command continue at the address that
 *     follows the previous burst (INCR semantics preserved across splits).
 *  4. Beat-counting integrity: every command eventually receives EXACTLY
 *     length+1 R beats - none dropped, none duplicated (the Mnist bug class).
 */
class DMAReaderFormal extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 0)
  // maxBurstBeats = 4 shrinks the burst-splitting loop so chained bursts are
  // reachable within a handful of BMC steps (256 would need 256+ cycles).
  val dut = new DMAReader(UInt(8 bits), shape = Seq(4), outLanes = 4, axiConfig, maxBurstBeats = 4)

  anyseq(dut.io.cmd.valid)
  anyseq(dut.io.cmd.payload)
  anyseq(dut.io.axiMaster.ar.ready)
  anyseq(dut.io.axiMaster.r.valid)
  anyseq(dut.io.axiMaster.r.payload)
  anyseq(dut.io.outStream.stream.ready)

  // Internal state handles
  val remaining     = dut.remaining.pull()
  val burstRemain   = dut.burstRemain.pull()
  val burstLen      = dut.burstLen.pull()
  val beatsToBnd    = dut.beatsToBoundary.pull()
  val addrRegH      = dut.addrReg.pull()

  assumeInitial(clockDomain.isResetActive)

  // ==========================================
  // ENVIRONMENT ASSUMPTIONS
  // ==========================================
  // AXI4 requires INCR burst start addresses to be aligned to the transfer
  // size (ar.size); Sequential only ever issues beat-aligned region offsets.
  assume(dut.io.cmd.payload.address(0, 2 bits) === 0)
  // The AXI slave only asserts r.valid for beats belonging to an outstanding
  // burst (no spurious responses), and obeys stable-valid handshakes.
  when(dut.io.axiMaster.r.valid) {
    assume(burstRemain =/= 0)
  }
  when(pastValid() && past(dut.io.cmd.valid) && !past(dut.io.cmd.ready)) {
    assume(dut.io.cmd.valid)
    assume(dut.io.cmd.payload === past(dut.io.cmd.payload))
  }
  when(pastValid() && past(dut.io.axiMaster.r.valid) && !past(dut.io.axiMaster.r.ready)) {
    assume(dut.io.axiMaster.r.valid)
    assume(dut.io.axiMaster.r.payload === past(dut.io.axiMaster.r.payload))
  }

  // ==========================================
  // 1. STRUCTURAL PROPERTIES
  // ==========================================
  when(pastValid()) {
    assert(dut.io.cmd.ready === ((remaining === 0) && (burstRemain === 0)))
    assert(dut.io.axiMaster.ar.valid === ((remaining =/= 0) && (burstRemain === 0)))
    // Downstream backpressure passes straight through the gearbox
    assert(dut.io.axiMaster.r.ready === dut.io.outStream.stream.ready)
    when(burstRemain =/= 0) {
      assert(dut.io.outStream.stream.valid === dut.io.axiMaster.r.valid)
    }
  }

  // ==========================================
  // 2. BURST LEGALITY
  // ==========================================
  when(dut.io.axiMaster.ar.valid) {
    assert(dut.io.axiMaster.ar.addr === addrRegH)
    assert(dut.io.axiMaster.ar.len === (burstLen - 1).resize(8 bits))
    assert(burstLen <= remaining)
    assert(burstLen >= 1)
    // No burst may cross a 4 KiB boundary (AXI4 protocol rule)
    assert((dut.io.axiMaster.ar.len.expand + 1) <= beatsToBnd)
  }

  // ==========================================
  // 3. CHAINED-BURST CONTIGUITY
  // ==========================================
  val arIdx = Reg(UInt(8 bits)) init (0)
  when(dut.io.cmd.fire) {
    arIdx := 0
  } elsewhen (dut.io.axiMaster.ar.fire) {
    arIdx := arIdx + 1
  }
  when(pastValid() && past(dut.io.axiMaster.ar.fire) && dut.io.axiMaster.ar.fire && past(arIdx =/= 0)) {
    val prevAddr = past(dut.io.axiMaster.ar.addr)
    val prevBeats = past(dut.io.axiMaster.ar.len).expand + 1
    assert(dut.io.axiMaster.ar.addr === (prevAddr + prevBeats * 4).resized)
  }

  // ==========================================
  // 4. BEAT-COUNTING INTEGRITY (per command)
  // ==========================================
  val busy   = RegInit(False)
  val expect = Reg(UInt(17 bits)) init (0)
  val got    = Reg(UInt(17 bits)) init (0)

  // Each accepted command re-arms the accounting (back-to-back commands are
  // legal the very cycle the previous one drains; cmd.fire cannot coincide
  // with an r.handshake since both require disjoint burst states).
  when(dut.io.cmd.fire) {
    expect := (dut.io.cmd.length +^ 1).resized
    got := 0
    busy := True
  } elsewhen (busy && dut.io.axiMaster.r.valid && dut.io.axiMaster.r.ready) {
    got := got + 1
  }
  when(busy && (got === expect) && (remaining === 0) && (burstRemain === 0)) {
    busy := False
  }

  when(pastValid() && busy) {
    assert(got <= expect, "over-delivery of R beats")
    when(remaining === 0 && burstRemain === 0) {
      assert(got === expect, "drained before all requested beats delivered")
    }
  }
}

object DMAReaderFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(180)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAReaderFormal, "dma_reader_formal")
  }
}
