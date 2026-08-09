package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Transpose
case class TransposeTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I4(), Seq(2, 3), lanes = 1)) // 2x3 matrix
    val c = master(Tensor(I4(), Seq(3, 2), lanes = 1)) // transposed to 3x2
  }
  
  io.c <> spinalML.ops.transpose(io.a)
}

class TransposeTest extends AnyFunSuite {
  test("Test streaming Transpose operation on 2x3 matrix") {
    SimConfig.withWave.compile(TransposeTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed A (2x3 matrix: [1, 2, 3], [4, 5, 6])
      val inputMatrix = Array(1, 2, 3, 4, 5, 6)
      
      fork {
        for(v <- inputMatrix) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= v
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        }
        dut.io.a.stream.valid #= false
      }
      
      // Check results (Transposed 3x2 matrix: [1, 4], [2, 5], [3, 6])
      val expectedOutput = Array(1, 4, 2, 5, 3, 6)
      
      for(v <- expectedOutput) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
        assert(dut.io.c.stream.payload(0).toInt == v, s"Expected $v but got ${dut.io.c.stream.payload(0).toInt}")
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Transpose compilation on FP4") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP4_E2M1(), Seq(2, 3), lanes = 1))
      val c = master(Tensor(FP4_E2M1(), Seq(3, 2), lanes = 1))
      c <> spinalML.ops.transpose(a)
    })
  }
}
