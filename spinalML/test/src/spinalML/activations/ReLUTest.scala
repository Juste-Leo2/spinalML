package spinalML.activations

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the ReLU operation
case class ReLUTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4), lanes = 2))
    val y = master(Tensor(dataType, Seq(4), lanes = 2))
  }
  
  io.y <> spinalML.activations.relu(io.x)
}

class ReLUTest extends AnyFunSuite {
  test("Test streaming ReLU operation on I4 tensors") {
    SimConfig.withWave.compile(ReLUTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.x.stream.valid #= true
      dut.io.x.stream.payload(0) #= 5
      dut.io.x.stream.payload(1) #= -5
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 1
      assert(dut.io.y.stream.payload(0).toInt == 5)
      assert(dut.io.y.stream.payload(1).toInt == 0)
      
      // Send second chunk
      dut.io.x.stream.payload(0) #= -6
      dut.io.x.stream.payload(1) #= 3
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 2
      assert(dut.io.y.stream.payload(0).toInt == 0)
      assert(dut.io.y.stream.payload(1).toInt == 3)
      
      dut.io.x.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test ReLU compilation on I16") {
    SpinalConfig().generateVerilog(ReLUTestComp(I16()))
  }

  test("Test ReLU compilation on FP4") {
    SpinalConfig().generateVerilog(ReLUTestComp(FP4_E2M1()))
  }

  test("Test ReLU compilation on BF16") {
    SpinalConfig().generateVerilog(ReLUTestComp(BF16()))
  }
}
