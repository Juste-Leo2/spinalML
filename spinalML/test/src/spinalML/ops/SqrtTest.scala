package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}

case class SqrtTestComp[T <: Data](dataType: HardType[T], forceAlg: Boolean = false) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> sqrt(io.a, forceAlg)
}

class SqrtTest extends AnyFunSuite {
  test("Sqrt compilation on I8") { SpinalConfig().generateVerilog(SqrtTestComp(I8())) }
  test("Sqrt compilation on I16") { SpinalConfig().generateVerilog(SqrtTestComp(I16())) }
  test("Sqrt compilation on FP8") { SpinalConfig().generateVerilog(SqrtTestComp(FP8_E4M3())) }
  test("Sqrt compilation on BF16") { SpinalConfig().generateVerilog(SqrtTestComp(BF16())) }
}
