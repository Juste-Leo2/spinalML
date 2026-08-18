package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib.bus.amba4.axi._
import spinalML.memory.{DMAReader, FetchRequest}

class DMAReaderFormal extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 0)
  val dut = new DMAReader(UInt(8 bits), shape = Seq(4), outLanes = 4, axiConfig)
  
  anyseq(dut.io.cmd.valid)
  anyseq(dut.io.cmd.payload)
  anyseq(dut.io.axiMaster.ar.ready)
  anyseq(dut.io.axiMaster.r.valid)
  anyseq(dut.io.axiMaster.r.payload)
  anyseq(dut.io.outStream.stream.ready)
  
  // ==========================================
  // STRUCTURAL & SAFETY PROPERTIES
  // ==========================================
  
  // 1. Command Passthrough (AR Channel)
  when(pastValid()) {
    assert(dut.io.axiMaster.ar.valid === dut.io.cmd.valid)
    assert(dut.io.cmd.ready === dut.io.axiMaster.ar.ready)
    assert(dut.io.axiMaster.ar.addr === dut.io.cmd.address)
    assert(dut.io.axiMaster.ar.len === dut.io.cmd.length)
  }
  
  // 2. Data Passthrough (R Channel mapped to Repack mapped to outStream)
  // Since repack is combinatorial here (axiLanes == outLanes == 4),
  // ready should exactly pass through.
  when(pastValid()) {
    assert(dut.io.axiMaster.r.ready === dut.io.outStream.stream.ready)
    assert(dut.io.axiMaster.r.valid === dut.io.outStream.stream.valid)
  }
}

object DMAReaderFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DMAReaderFormal, "dma_reader_formal")
  }
}
