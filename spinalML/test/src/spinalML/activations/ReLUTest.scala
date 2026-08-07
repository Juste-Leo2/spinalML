package spinalML.activations

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the ReLU operation
case class ReLUTestComp() extends Component {
  val io = new Bundle {
    val x = slave(Tensor(I8(), Seq(4), lanes = 2))
    val y = master(Tensor(I8(), Seq(4), lanes = 2))
  }
  
  io.y <> spinalML.activations.relu(io.x)
}

class ReLUTest extends AnyFunSuite {
  test("Test streaming ReLU operation on I8 tensors") {
    SimConfig.withWave.compile(ReLUTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.x.stream.valid #= true
      dut.io.x.stream.payload(0) #= 10
      dut.io.x.stream.payload(1) #= -5
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 1
      assert(dut.io.y.stream.payload(0).toInt == 10)
      assert(dut.io.y.stream.payload(1).toInt == 0)
      
      // Send second chunk
      dut.io.x.stream.payload(0) #= -12
      dut.io.x.stream.payload(1) #= 3
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 2
      assert(dut.io.y.stream.payload(0).toInt == 0)
      assert(dut.io.y.stream.payload(1).toInt == 3)
      
      dut.io.x.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
