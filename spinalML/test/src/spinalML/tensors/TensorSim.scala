package spinalML.tensors

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes._

class TensorTestModule extends Component {
  val io = new Bundle {
    val t2d = out(Tensor.Tensor2D(I8(), ne0 = 2, ne1 = 3))
    val t1d = out(Tensor.Tensor1D(U4(), ne0 = 4))
  }

  // Assign arbitrary values to the 2D tensor (I8)
  io.t2d(0, 0) := 10
  io.t2d(0, 1) := -5
  io.t2d(0, 2) := 20
  io.t2d(1, 0) := 0
  io.t2d(1, 1) := 127
  io.t2d(1, 2) := -128

  // Assign arbitrary values to the 1D tensor (U4)
  io.t1d(0) := 0
  io.t1d(1) := 5
  io.t1d(2) := 10
  io.t1d(3) := 15
}

class TensorSim extends AnyFunSuite {
  test("Tensor indexing and instantiation") {
    SimConfig.withWave.compile(new TensorTestModule).doSim { dut =>
      // Wait for combinational logic
      sleep(1)

      // Test 2D Tensor (I8) values
      assert(dut.io.t2d(0, 0).toInt == 10)
      assert(dut.io.t2d(0, 1).toInt == -5)
      assert(dut.io.t2d(0, 2).toInt == 20)
      assert(dut.io.t2d(1, 0).toInt == 0)
      assert(dut.io.t2d(1, 1).toInt == 127)
      assert(dut.io.t2d(1, 2).toInt == -128)

      // Test 1D Tensor (U4) values
      assert(dut.io.t1d(0).toInt == 0)
      assert(dut.io.t1d(1).toInt == 5)
      assert(dut.io.t1d(2).toInt == 10)
      assert(dut.io.t1d(3).toInt == 15)

      // Verify the flat index logic internally
      assert(dut.io.t2d.getFlatIndex(Seq(1, 2)) == 5) // (1*3 + 2)
    }
  }
}
