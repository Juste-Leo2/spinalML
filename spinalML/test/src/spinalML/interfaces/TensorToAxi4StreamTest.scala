package spinalML.interfaces

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib.bus.amba4.axis._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

class TensorToAxi4StreamTest extends AnyFunSuite {
  test("Test conversion from Tensor stream to standard AXI4-Stream") {
    // We create a Component dynamically to test the converter
    // A tensor of shape (4) with 2 lanes (so it takes 2 chunks to send)
    SimConfig.withWave.compile(TensorToAxi4Stream(I8(), Seq(4), lanes = 2, axiDataWidth = 32)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Init
      dut.io.tensor.stream.valid #= false
      dut.io.axis.ready #= true
      dut.clockDomain.waitSampling()
      
      var seenData = scala.collection.mutable.ArrayBuffer[BigInt]()
      var seenLast = scala.collection.mutable.ArrayBuffer[Boolean]()
      
      StreamMonitor(dut.io.axis, dut.clockDomain) { payload =>
        seenData += payload.data.toBigInt
        seenLast += payload.last.toBoolean
      }
      
      // Send chunk 1: 5 and -10
      dut.io.tensor.stream.valid #= true
      dut.io.tensor.stream.payload(0) #= 5
      dut.io.tensor.stream.payload(1) #= -10
      dut.clockDomain.waitSamplingWhere(dut.io.tensor.stream.ready.toBoolean)
      
      // Send chunk 2: 20 and 42
      dut.io.tensor.stream.payload(0) #= 20
      dut.io.tensor.stream.payload(1) #= 42
      dut.clockDomain.waitSamplingWhere(dut.io.tensor.stream.ready.toBoolean)
      
      dut.io.tensor.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      // Verify first chunk
      // lane 0 = 5 (0x05), lane 1 = -10 (0xF6) => Packed = 0xF605
      assert(seenData(0) == 0xF605)
      assert(seenLast(0) == false) // First chunk of 2, so not last
      
      // Verify second chunk
      // lane 0 = 20 (0x14), lane 1 = 42 (0x2A) => Packed = 0x2A14
      assert(seenData(1) == 0x2A14)
      assert(seenLast(1) == true) // Second chunk is the last!
    }
  }
}
