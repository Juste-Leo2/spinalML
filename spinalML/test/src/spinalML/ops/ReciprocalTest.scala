package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1, I16, BF16}

case class ReciprocalTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> reciprocal(io.a)
}

class ReciprocalTest extends AnyFunSuite {
  test("Reciprocal LUT compilation on I4") { SpinalConfig().generateVerilog(ReciprocalTestComp(I4())) }
  test("Reciprocal LUT compilation on FP4") { SpinalConfig().generateVerilog(ReciprocalTestComp(FP4_E2M1())) }
  test("Reciprocal PWL compilation on I16") { SpinalConfig().generateVerilog(ReciprocalTestComp(I16())) }
  test("Reciprocal PWL compilation on BF16") { SpinalConfig().generateVerilog(ReciprocalTestComp(BF16())) }
}
