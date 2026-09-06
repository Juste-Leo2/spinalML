// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class SoftmaxTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(16, 4), lanes = 4))
    val y = master(Tensor(dataType, Seq(16, 4), lanes = 4))
  }
  
  val comp = Softmax1D(dataType, channels = 4, seqLen = 16)
  comp.io.x <> io.x
  io.y <> comp.io.y
}

// Non-power-of-2 channel softmax (Python/Cocotb toplevel for the 10-channel golden).
case class SoftmaxChannelsTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4, 10), lanes = 10))
    val y = master(Tensor(dataType, Seq(4, 10), lanes = 10))
  }

  val comp = Softmax1D(dataType, channels = 10, seqLen = 4)
  comp.io.x <> io.x
  io.y <> comp.io.y
}

// Odd channel counts exercised at elaboration (only used by the Scala suite).
case class SoftmaxOddChannelsTestComp[T <: Data](dataType: HardType[T], channels: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4, channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(4, channels), lanes = channels))
  }

  val comp = Softmax1D(dataType, channels = channels, seqLen = 4)
  comp.io.x <> io.x
  io.y <> comp.io.y
}

class SoftmaxTest extends AnyFunSuite {
  test("Softmax1D compilation on I8") { SpinalConfig().generateVerilog(SoftmaxTestComp(I8())) }
  test("Softmax1D compilation on I16") { SpinalConfig().generateVerilog(SoftmaxTestComp(I16())) }
  test("Softmax1D compilation on FP8") { SpinalConfig().generateVerilog(SoftmaxTestComp(FP8_E4M3())) }
  test("Softmax1D compilation on BF16") { SpinalConfig().generateVerilog(SoftmaxTestComp(BF16())) }

  test("Softmax1D arbitrary channels (10) compilation on I8") { SpinalConfig().generateVerilog(SoftmaxChannelsTestComp(I8())) }
  test("Softmax1D arbitrary channels (10) compilation on I16") { SpinalConfig().generateVerilog(SoftmaxChannelsTestComp(I16())) }
  test("Softmax1D arbitrary channels (10) compilation on FP8") { SpinalConfig().generateVerilog(SoftmaxChannelsTestComp(FP8_E4M3())) }
  test("Softmax1D arbitrary channels (10) compilation on BF16") { SpinalConfig().generateVerilog(SoftmaxChannelsTestComp(BF16())) }

  test("Softmax1D odd channels (3) compilation on I8") { SpinalConfig().generateVerilog(SoftmaxOddChannelsTestComp(I8(), 3)) }
  test("Softmax1D odd channels (3) compilation on I16") { SpinalConfig().generateVerilog(SoftmaxOddChannelsTestComp(I16(), 3)) }
  test("Softmax1D odd channels (3) compilation on FP8") { SpinalConfig().generateVerilog(SoftmaxOddChannelsTestComp(FP8_E4M3(), 3)) }
  test("Softmax1D odd channels (3) compilation on BF16") { SpinalConfig().generateVerilog(SoftmaxOddChannelsTestComp(BF16(), 3)) }
}
