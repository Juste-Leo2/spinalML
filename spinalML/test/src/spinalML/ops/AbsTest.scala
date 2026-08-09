package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3}

class AbsTest extends AnyFunSuite {
  test("Abs compilation on I8") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(I8(), Seq(2, 2), lanes = 2))
      val c = master(Tensor(I8(), Seq(2, 2), lanes = 2))
      c <> abs(a)
    })
  }

  test("Abs compilation on FP8") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP8_E4M3(), Seq(2, 2), lanes = 2))
      val c = master(Tensor(FP8_E4M3(), Seq(2, 2), lanes = 2))
      c <> abs(a)
    })
  }
}
