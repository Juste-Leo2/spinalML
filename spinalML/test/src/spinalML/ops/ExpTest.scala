package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}

case class ExpTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> exp(io.a)
}

class ExpTest extends AnyFunSuite {
  test("Exp LUT compilation on I4") {
    SpinalConfig().generateVerilog(ExpTestComp(I4()))
  }

  test("Exp LUT compilation on FP4") {
    SpinalConfig().generateVerilog(ExpTestComp(FP4_E2M1()))
  }
}
