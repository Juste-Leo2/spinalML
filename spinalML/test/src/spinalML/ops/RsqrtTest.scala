package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}

case class RsqrtTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> rsqrt(io.a)
}

class RsqrtTest extends AnyFunSuite {
  test("Rsqrt LUT compilation on I4") {
    SpinalConfig().generateVerilog(RsqrtTestComp(I4()))
  }

  test("Rsqrt LUT compilation on FP4") {
    SpinalConfig().generateVerilog(RsqrtTestComp(FP4_E2M1()))
  }

  test("Rsqrt PWL compilation on I16") {
    SpinalConfig().generateVerilog(RsqrtTestComp(spinalML.dtypes.I16()))
  }

  test("Rsqrt PWL compilation on BF16") {
    SpinalConfig().generateVerilog(RsqrtTestComp(spinalML.dtypes.BF16()))
  }
}
