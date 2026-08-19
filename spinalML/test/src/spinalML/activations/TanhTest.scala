package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class TanhTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> tanh(io.a)
}

class TanhTest extends AnyFunSuite {
  test("Tanh compilation on I8") {
    SpinalConfig().generateVerilog(TanhTestComp(I8()))
  }

  test("Tanh compilation on FP8") {
    SpinalConfig().generateVerilog(TanhTestComp(FP8_E4M3()))
  }

  test("Tanh compilation on I16") {
    SpinalConfig().generateVerilog(TanhTestComp(spinalML.dtypes.I16()))
  }

  test("Tanh compilation on BF16") {
    SpinalConfig().generateVerilog(TanhTestComp(spinalML.dtypes.BF16()))
  }
}