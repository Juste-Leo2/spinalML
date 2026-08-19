package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class LogTestComp[T <: Data](dataType: HardType[T], base: Double = Math.E) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> log(io.a, base)
}

case class LogTestComp10[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> log(io.a, 10.0)
}

class LogTest extends AnyFunSuite {
  test("Log LUT compilation on I8") {
    SpinalConfig().generateVerilog(LogTestComp(I8()))
  }

  test("Log LUT compilation on FP8") {
    SpinalConfig().generateVerilog(LogTestComp(FP8_E4M3()))
  }

  test("Log PWL compilation on I16") {
    SpinalConfig().generateVerilog(LogTestComp(spinalML.dtypes.I16()))
  }

  test("Log Alg+LUT compilation on BF16 default") {
    SpinalConfig().generateVerilog(LogTestComp(spinalML.dtypes.BF16()))
  }

  test("Log Alg+LUT compilation on BF16 base10") {
    SpinalConfig().generateVerilog(LogTestComp10(spinalML.dtypes.BF16()))
  }
}