package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1, I16, BF16}

case class SqrtTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> sqrt(io.a)
}

class SqrtTest extends AnyFunSuite {
  test("Sqrt LUT compilation on I4") { SpinalConfig().generateVerilog(SqrtTestComp(I4())) }
  test("Sqrt LUT compilation on FP4") { SpinalConfig().generateVerilog(SqrtTestComp(FP4_E2M1())) }
  test("Sqrt PWL compilation on I16") { SpinalConfig().generateVerilog(SqrtTestComp(I16())) }
  test("Sqrt PWL compilation on BF16") { SpinalConfig().generateVerilog(SqrtTestComp(BF16())) }
}
