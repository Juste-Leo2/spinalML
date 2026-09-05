// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}

case class ReciprocalTestComp[T <: Data](dataType: HardType[T], forceAlg: Boolean = false) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> reciprocal(io.a, forceAlg)
}

class ReciprocalTest extends AnyFunSuite {
  test("Reciprocal compilation on I8") { SpinalConfig().generateVerilog(ReciprocalTestComp(I8())) }
  test("Reciprocal compilation on I16") { SpinalConfig().generateVerilog(ReciprocalTestComp(I16())) }
  test("Reciprocal compilation on FP8") { SpinalConfig().generateVerilog(ReciprocalTestComp(FP8_E4M3())) }
  test("Reciprocal compilation on BF16") { SpinalConfig().generateVerilog(ReciprocalTestComp(BF16())) }
}
