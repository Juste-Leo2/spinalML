package spinalML.activations

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the LeakyReLU operation
case class LeakyReLUTestComp() extends Component {
  val io = new Bundle {
    val x = slave(Tensor(I8(), Seq(4), lanes = 2))
    val y = master(Tensor(I8(), Seq(4), lanes = 2))
  }
  
  // shift = 2 means multiplying by 0.25
  io.y <> spinalML.activations.leaky_relu(io.x, shift = 2)
}

class LeakyReLUTest extends AnyFunSuite {
  test("Test streaming LeakyReLU operation on I8 tensors") {
    SimConfig.withWave.compile(LeakyReLUTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.x.stream.valid #= true
      dut.io.x.stream.payload(0) #= 10
      dut.io.x.stream.payload(1) #= -8
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 1
      assert(dut.io.y.stream.payload(0).toInt == 10)
      // -8 >> 2 = -2
      assert(dut.io.y.stream.payload(1).toInt == -2)
      
      // Send second chunk
      dut.io.x.stream.payload(0) #= -12
      dut.io.x.stream.payload(1) #= 3
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 2
      // -12 >> 2 = -3
      assert(dut.io.y.stream.payload(0).toInt == -3)
      assert(dut.io.y.stream.payload(1).toInt == 3)
      
      dut.io.x.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
