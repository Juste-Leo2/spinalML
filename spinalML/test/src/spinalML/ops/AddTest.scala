package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the add operation
case class AddTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(4), lanes = 2))
    val b = slave(Tensor(I8(), Seq(4), lanes = 2))
    val c = master(Tensor(I8(), Seq(4), lanes = 2))
  }
  
  // GGML-like syntax for add
  io.c <> spinalML.ops.add(io.a, io.b)
}

class AddTest extends AnyFunSuite {
  test("Test streaming add operation on I8 tensors") {
    SimConfig.withWave.compile(AddTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 10
      dut.io.a.stream.payload(1) #= -5
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 15
      dut.io.b.stream.payload(1) #= 10
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Check results for chunk 1
      assert(dut.io.c.stream.payload(0).toInt == 25)
      assert(dut.io.c.stream.payload(1).toInt == 5)
      
      // Send second chunk
      dut.io.a.stream.payload(0) #= 2
      dut.io.a.stream.payload(1) #= 3
      
      dut.io.b.stream.payload(0) #= 4
      dut.io.b.stream.payload(1) #= 6
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Check results for chunk 2
      assert(dut.io.c.stream.payload(0).toInt == 6)
      assert(dut.io.c.stream.payload(1).toInt == 9)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
