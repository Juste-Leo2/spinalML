package spinalML.poolings

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the AvgPool1D operation
case class AvgPool1DTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 1), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 1), lanes = 1))
  }
  
  // poolSize = 2, stride = 2
  io.c <> spinalML.poolings.avgpool1d(io.a, poolSize = 2, stride = 2)
}

class AvgPool1DTest extends AnyFunSuite {
  test("Test streaming AvgPool1D operation on I8 tensors") {
    SimConfig.withWave.compile(AvgPool1DTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send sequence: 5, 9, 2, 4
      val sequence = Array(5, 9, 2, 4)
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
      // avg(5, 9) = 14 >> 1 = 7
      assert(dut.io.c.stream.payload(0).toInt == 7)
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      // avg(2, 4) = 6 >> 1 = 3
      assert(dut.io.c.stream.payload(0).toInt == 3)
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test AvgPool1D compilation on I8") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(I8()))
  }

  test("Test AvgPool1D compilation on FP8") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(FP8_E4M3()))
  }

  test("Test AvgPool1D compilation on I16") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(I16()))
  }

  test("Test AvgPool1D compilation on BF16") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(BF16()))
  }
}
