// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class SigmoidTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> sigmoid(io.a)
}

class SigmoidTest extends AnyFunSuite {
  test("Sigmoid compilation on I8") {
    SpinalConfig().generateVerilog(SigmoidTestComp(I8()))
  }

  test("Sigmoid compilation on FP8") {
    SpinalConfig().generateVerilog(SigmoidTestComp(FP8_E4M3()))
  }

  test("Sigmoid compilation on I16") {
    SpinalConfig().generateVerilog(SigmoidTestComp(spinalML.dtypes.I16()))
  }

  test("Sigmoid compilation on BF16") {
    SpinalConfig().generateVerilog(SigmoidTestComp(spinalML.dtypes.BF16()))
  }
}