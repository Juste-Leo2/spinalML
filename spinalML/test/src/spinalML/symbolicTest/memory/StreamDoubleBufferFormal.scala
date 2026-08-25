package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.StreamDoubleBuffer

class StreamDoubleBufferFormal extends Component {
  val dut = new StreamDoubleBuffer(UInt(8 bits), depth = 4, lanes = 1)
  
  // Inject random inputs
  anyseq(dut.io.streamIn.valid)
  anyseq(dut.io.streamIn.payload)
  anyseq(dut.io.readAddr)
  anyseq(dut.io.nextTile)
  // One-shot contract: no command boundary inside the formal window.
  dut.io.reArm := False
  
  // Pull internal signals across hierarchy to avoid violations
  val pingFull = dut.pingFull.pull()
  val pongFull = dut.pongFull.pull()
  val computeBank = dut.computeBank.pull()
  val loadCounterValue = dut.loadCounter.value.pull()
  
  // ==========================================
  // SAFETY PROPERTIES (No data loss or corruption)
  // ==========================================
  
  // 1. Backpressure Check: If both banks are full, the stream must NOT be ready to accept data.
  // This prevents overwriting data that hasn't been consumed.
  assert(!(pingFull && pongFull && dut.io.streamIn.ready))
  
  // 2. Compute View Check: The compute interface should only see tileReady if the computeBank is actually full.
  val expectedTileReady = (computeBank === False) ? pingFull | pongFull
  assert(dut.io.tileReady === expectedTileReady)
  
  // 3. (Removed tautological bounds check, Counter(4) is natively 2-bits and cannot exceed 3).
  
  // ==========================================
  // REACHABILITY (Liveness / No deadlock)
  // ==========================================
  
  // 1. Can we reach a state where both banks are full? (Meaning the stream can run ahead of compute)
  cover(pingFull && pongFull)
  
  // 2. Can we successfully consume a tile?
  cover(dut.io.nextTile && dut.io.tileReady)
}

object StreamDoubleBufferFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15) // We check up to 15 cycles since it takes 4 cycles to fill one bank
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new StreamDoubleBufferFormal, "stream_double_buffer_formal")
  }
}
