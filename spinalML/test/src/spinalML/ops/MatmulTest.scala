package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing matmul: Matrix A [1, 2] x Vector B [2, 1]
case class MatmulTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2, 1), lanes = 2))
    val c = master(Tensor(dataType, Seq(1, 1), lanes = 1))
  }
  
  // tileSize = 2
  io.c <> spinalML.ops.matmul(io.a, io.b, tileSize = 2)
}

class MatmulTest extends AnyFunSuite {
  test("Test streaming matmul (SRAM + MAC) operation on I4 tensors") {
    SimConfig.withWave.compile(MatmulTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load matrix B into internal SRAM
      // B = [3, -2]T
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= -2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.valid #= false
      
      // Step 2: Stream Matrix A to compute
      // Row 0: [2, 1]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 2
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 3: Wait for output C
      // 2*3 + 1*(-2) = 6 - 2 = 4
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 4)
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Matmul compilation on I16") {
    SpinalConfig().generateVerilog(MatmulTestComp(I16()))
  }

  test("Test Matmul compilation on FP4") {
    SpinalConfig().generateVerilog(MatmulTestComp(FP4_E2M1()))
  }

  test("Test Matmul compilation on BF16") {
    SpinalConfig().generateVerilog(MatmulTestComp(BF16()))
  }
}
