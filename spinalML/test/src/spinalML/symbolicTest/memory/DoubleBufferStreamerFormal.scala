// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.DoubleBufferStreamer

class DoubleBufferStreamerFormal extends Component {
  val dut = FormalDut(DoubleBufferStreamer(UInt(8 bits), depth = 4, lanes = 1))
  
  // Drive inputs
  anyseq(dut.io.tileReady)
  anyseq(dut.io.readData)
  anyseq(dut.io.streamOut.ready)
  
  // Internal signals
  val isReading = dut.isReading.pull()
  val readCounter = dut.readCounter.value.pull()
  val delayedValid = dut.delayedValid.pull()
  val fifoPushReady = dut.fifo.io.push.ready.pull()
  val reqStreamFire = dut.reqStream.fire.pull()
  val willOverflow = dut.readCounter.willOverflowIfInc.pull()
  
  // Start from reset
  assumeInitial(clockDomain.isResetActive)
  
  when(pastValid()) {
    // 1. Safe FIFO Access
    // When delayedValid is true (we are pushing data), the FIFO must be ready
    when(delayedValid) {
      assert(fifoPushReady, "FIFO overflow! Tried to push when FIFO was not ready.")
    }
    
    // 2. Read Address Bounds (Removed tautological bounds check since log2Up(4) is natively 2-bits and cannot exceed 3).
    
    // 3. Sequential Increment Check
    // If we are reading and we fire a request, the counter should increment (unless overflowing)
    when(past(isReading) && past(reqStreamFire)) {
      when(past(willOverflow)) {
        assert(!isReading, "isReading should drop when counter overflows")
      } elsewhen(!clockDomain.isResetActive) {
        assert(readCounter === past(readCounter) + 1, "Counter did not increment linearly")
      }
    }
    
    // 4. Handshake NextTile logic
    // nextTile is combinatorial and should only be asserted when the counter overflows this cycle
    when(dut.io.nextTile) {
      assert(isReading && reqStreamFire && willOverflow, "nextTile asserted spuriously!")
    }
  }
}

object DoubleBufferStreamerFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DoubleBufferStreamerFormal, "double_buffer_streamer_formal")
  }
}
