package spinalML.poolings

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the MaxPool1D operation
case class MaxPool1DTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 1), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 1), lanes = 1))
  }
  
  // poolSize = 2, stride = 2
  io.c <> spinalML.poolings.maxpool1d(io.a, poolSize = 2, stride = 2)
}

class MaxPool1DTest extends AnyFunSuite {
  test("Test streaming MaxPool1D operation on I4 tensors") {
    SimConfig.withWave.compile(MaxPool1DTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send sequence: 5, -2, -3, 4
      val sequence = Array(5, -2, -3, 4)
      var i = 0
      
      // Feed data
      fork {
        while (i < sequence.length) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= sequence(i)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          i += 1
        }
        dut.io.a.stream.valid #= false
      }
      
      // Check results
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 5) // max(5, -2)
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 4) // max(-3, 4)
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test MaxPool1D compilation on I16") {
    SpinalConfig().generateVerilog(MaxPool1DTestComp(I16()))
  }

  test("Test MaxPool1D compilation on FP4") {
    SpinalConfig().generateVerilog(MaxPool1DTestComp(FP4_E2M1()))
  }

  test("Test MaxPool1D compilation on BF16") {
    SpinalConfig().generateVerilog(MaxPool1DTestComp(BF16()))
  }
}
