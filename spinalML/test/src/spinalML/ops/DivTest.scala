// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class DivTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> div(io.a, io.b)
}

class DivTest extends AnyFunSuite {
  test("Div LUT compilation on I8") { SpinalConfig().generateVerilog(DivTestComp(I8())) }
  test("Div LUT compilation on FP8") { SpinalConfig().generateVerilog(DivTestComp(FP8_E4M3())) }
  test("Div PWL compilation on I16") { SpinalConfig().generateVerilog(DivTestComp(I16())) }
  test("Div PWL compilation on BF16") { SpinalConfig().generateVerilog(DivTestComp(BF16())) }
}
