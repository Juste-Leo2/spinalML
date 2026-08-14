package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Transpose 2x3 -> 3x2
case class TransposeTestComp_2x3[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 3), lanes = 1))
    val c = master(Tensor(dataType, Seq(3, 2), lanes = 1))
  }
  io.c <> spinalML.ops.transpose(io.a)
}

// Component for testing Transpose 4x4 -> 4x4
case class TransposeTestComp_4x4[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 4), lanes = 1))
    val c = master(Tensor(dataType, Seq(4, 4), lanes = 1))
  }
  io.c <> spinalML.ops.transpose(io.a)
}

class TransposeTest extends AnyFunSuite {
  test("Test streaming Transpose operation on 2x3 matrix") {
    SimConfig.withWave.compile(TransposeTestComp_2x3(I8())).doSim { dut =>
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

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test Transpose compilation on $name") {
      SpinalConfig().generateVerilog(TransposeTestComp_2x3(dt()))
      SpinalConfig().generateVerilog(TransposeTestComp_4x4(dt()))
    }
  }
}
