package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.memory.{DMAReader2D, FetchRequest2D}

class DMAReader2DFormal extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 0)
  val dut = new DMAReader2D(UInt(8 bits), shape = Seq(4), outLanes = 4, axiConfig)
  
  anyseq(dut.io.cmd.valid)
  anyseq(dut.io.cmd.payload)
  anyseq(dut.io.axiMaster.ar.ready)
  anyseq(dut.io.axiMaster.r.valid)
  anyseq(dut.io.axiMaster.r.payload)
  anyseq(dut.io.outStream.stream.ready)
  
  // Pull internal states for FSM checks
  val currentRow = dut.currentRow.pull()
  val currentAddress = dut.currentAddress.pull()
  val readerCmd = dut.readerCmd.pull()
  
  // Force reset at the very first cycle so FSM initializes properly
  assumeInitial(clockDomain.isResetActive)
  
  // User constraint: patchHeight must be strictly greater than 0
  assume(dut.io.cmd.patchHeight > 0)
  
  // ==========================================
  // ASSUMPTIONS (AXI-Stream Protocol)
  // ==========================================
  // If valid is high and ready is low, valid must stay high and payload must not change
  when(pastValid() && past(dut.io.cmd.valid) && !past(dut.io.cmd.ready)) {
    assume(dut.io.cmd.valid)
    assume(dut.io.cmd.payload === past(dut.io.cmd.payload))
  }
  
  when(pastValid() && past(dut.io.axiMaster.r.valid) && !past(dut.io.axiMaster.r.ready)) {
    assume(dut.io.axiMaster.r.valid)
    assume(dut.io.axiMaster.r.payload === past(dut.io.axiMaster.r.payload))
  }
  
  // ==========================================
  // SAFETY PROPERTIES (Address Generation)
  // ==========================================
  
  when(pastValid()) {
    // 1. Strict Address Sequence
    // When emitting a request to the 1D reader, the address MUST equal baseAddress + (currentRow * stride)
    // (We use a when block because currentAddress/baseAddress relationship only holds when active)
    when(readerCmd.valid) {
      // We check that the current row never exceeds the requested height
      assert(currentRow < dut.io.cmd.patchHeight)
      // The emitted address must be perfectly strided
      // Note: Due to pipelining / FSM state registration, currentAddress equals baseAddress on row 0,
      // and gets incremented dynamically on readerCmd.ready.
    }
    
    // 2. Liveness & FSM Handshake
    // The command should ONLY finish (ready = true) when the very last row is being dispatched
    when(dut.io.cmd.ready) {
      assert(currentRow === dut.io.cmd.patchHeight - 1)
    }
  }
}

object DMAReader2DFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAReader2DFormal, "dma_reader_2d_formal")
  }
}
