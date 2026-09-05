// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3}

case class ExpTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> exp(io.a)
}

class ExpTest extends AnyFunSuite {
  test("Exp LUT compilation on I8") {
    SpinalConfig().generateVerilog(ExpTestComp(I8()))
  }

  test("Exp LUT compilation on FP8") {
    SpinalConfig().generateVerilog(ExpTestComp(FP8_E4M3()))
  }

  test("Exp PWL compilation on I16") {
    SpinalConfig().generateVerilog(ExpTestComp(spinalML.dtypes.I16()))
  }

  test("Exp PWL compilation on BF16") {
    SpinalConfig().generateVerilog(ExpTestComp(spinalML.dtypes.BF16()))
  }
}
