package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Hardware component to test the subtract operation
case class SubTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4), lanes = 2))
    val c = master(Tensor(dataType, Seq(4), lanes = 2))
  }
  
  // Use the GGML-like syntax for subtraction
  io.c <> spinalML.ops.sub(io.a, io.b)
}

class SubTest extends AnyFunSuite {
  test("Test streaming sub operation on I4 tensors") {
    SimConfig.withWave.compile(SubTestComp(I4())).doSim { dut =>
      // Generate a clock with a period of 10 simulation units
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream handshake signals
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send the first chunk (2 elements per lane)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 5
      dut.io.a.stream.payload(1) #= -2
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= 4
      
      // Wait until the operation computes and outputs valid data
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Verify results for chunk 1 (5 - 3 = 2, -2 - 4 = -6)
      assert(dut.io.c.stream.payload(0).toInt == 2)
      assert(dut.io.c.stream.payload(1).toInt == -6)
      
      // Send the second chunk
      dut.io.a.stream.payload(0) #= 2
      dut.io.a.stream.payload(1) #= 3
      
      dut.io.b.stream.payload(0) #= 5
      dut.io.b.stream.payload(1) #= -4
      
      // Wait for output
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Verify results for chunk 2 (2 - 5 = -3, 3 - (-4) = 7)
      assert(dut.io.c.stream.payload(0).toInt == -3)
      assert(dut.io.c.stream.payload(1).toInt == 7)
      
      // Stop sending data
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Sub compilation on I16") {
    SpinalConfig().generateVerilog(SubTestComp(I16()))
  }

  test("Test Sub compilation on FP4") {
    SpinalConfig().generateVerilog(SubTestComp(FP4_E2M1()))
  }

  test("Test Sub compilation on BF16") {
    SpinalConfig().generateVerilog(SubTestComp(BF16()))
  }
}
