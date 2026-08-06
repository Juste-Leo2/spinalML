package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Hardware component to test the subtract operation
case class SubTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(4), lanes = 2))
    val b = slave(Tensor(I8(), Seq(4), lanes = 2))
    val c = master(Tensor(I8(), Seq(4), lanes = 2))
  }
  
  // Use the GGML-like syntax for subtraction
  io.c <> spinalML.ops.sub(io.a, io.b)
}

class SubTest extends AnyFunSuite {
  test("Test streaming sub operation on I8 tensors") {
    SimConfig.withWave.compile(SubTestComp()).doSim { dut =>
      // Generate a clock with a period of 10 simulation units
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream handshake signals
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send the first chunk (2 elements per lane)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 10
      dut.io.a.stream.payload(1) #= -5
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 15
      dut.io.b.stream.payload(1) #= 10
      
      // Wait until the operation computes and outputs valid data
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Verify results for chunk 1 (10 - 15 = -5, -5 - 10 = -15)
      assert(dut.io.c.stream.payload(0).toInt == -5)
      assert(dut.io.c.stream.payload(1).toInt == -15)
      
      // Send the second chunk
      dut.io.a.stream.payload(0) #= 20
      dut.io.a.stream.payload(1) #= 3
      
      dut.io.b.stream.payload(0) #= 5
      dut.io.b.stream.payload(1) #= 6
      
      // Wait for output
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Verify results for chunk 2 (20 - 5 = 15, 3 - 6 = -3)
      assert(dut.io.c.stream.payload(0).toInt == 15)
      assert(dut.io.c.stream.payload(1).toInt == -3)
      
      // Stop sending data
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
