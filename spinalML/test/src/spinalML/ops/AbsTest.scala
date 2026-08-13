package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}

case class AbsTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 2))
  }
  io.c <> abs(io.a)
}

class AbsTest extends AnyFunSuite {
  test("Abs compilation on I8") { SpinalConfig().generateVerilog(AbsTestComp(I8())) }
  test("Abs compilation on I16") { SpinalConfig().generateVerilog(AbsTestComp(I16())) }
  test("Abs compilation on FP8") { SpinalConfig().generateVerilog(AbsTestComp(FP8_E4M3())) }
  test("Abs compilation on BF16") { SpinalConfig().generateVerilog(AbsTestComp(BF16())) }
}
