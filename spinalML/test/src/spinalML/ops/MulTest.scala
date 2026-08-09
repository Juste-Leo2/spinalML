package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the mul operation
case class MulTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4), lanes = 2))
    val c = master(Tensor(dataType, Seq(4), lanes = 2))
  }
  
  // GGML-like syntax for mul
  io.c <> spinalML.ops.mul(io.a, io.b)
}

class MulTest extends AnyFunSuite {
  test("Test streaming pipelined mul operation on I4 tensors") {
    SimConfig.withWave.compile(MulTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 3
      dut.io.a.stream.payload(1) #= -2
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 2
      dut.io.b.stream.payload(1) #= 3
      
      // Because of m2sPipe, the result will take 1 cycle to arrive.
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Check results for chunk 1 (3*2 = 6, -2*3 = -6)
      assert(dut.io.c.stream.payload(0).toInt == 6)
      assert(dut.io.c.stream.payload(1).toInt == -6)
      
      // Stop input stream to ensure pipeline flushes properly
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.clockDomain.waitSampling(3)
    }
  }

  test("Test Mul compilation on I16") {
    SpinalConfig().generateVerilog(MulTestComp(I16()))
  }

  test("Test Mul compilation on FP4") {
    SpinalConfig().generateVerilog(MulTestComp(FP4_E2M1()))
  }

  test("Test Mul compilation on BF16") {
    SpinalConfig().generateVerilog(MulTestComp(BF16()))
  }
}
