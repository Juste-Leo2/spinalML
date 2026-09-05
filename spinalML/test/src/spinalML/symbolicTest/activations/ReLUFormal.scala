// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.activations

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.tensors.Tensor
import spinalML.activations.ReLUOp

case class ReLUTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(2), lanes = 2))
    val y = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  val relu = ReLUOp(dataType, Seq(2), 2)
  relu.io.x <> io.x
  io.y <> relu.io.y
}

class ReLUFormal_I8 extends Component {
  val dut = FormalDut(ReLUTestComp(SInt(8 bits)))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.x.stream.valid)
  assume(dut.io.y.stream.ready)
  
  val pastValidX = past(dut.io.x.stream.valid)
  val pastReadyX = past(dut.io.x.stream.ready)
  val pastPayloadX = past(dut.io.x.stream.payload)
  when(pastValidX && !pastReadyX) {
    assume(dut.io.x.stream.valid)
    assume(dut.io.x.stream.payload === pastPayloadX)
  }

  val expectedPayload = Vec(SInt(8 bits), 2)
  for(i <- 0 until 2) {
    val xVal = dut.io.x.stream.payload(i)
    val zero = SInt(8 bits)
    zero := 0
    expectedPayload(i) := Mux(xVal < 0, zero, xVal)
  }

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  
  when(fireOut) {
    for(i <- 0 until 2) {
      assert(dut.io.y.stream.payload(i) === expectedPayload(i), s"ReLU I8 mismatch on lane $i")
    }
  }
}

class ReLUFormal_FP4 extends Component {
  val dut = FormalDut(ReLUTestComp(FP4_E2M1()))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.x.stream.valid)
  assume(dut.io.y.stream.ready)
  
  val pastValidX = past(dut.io.x.stream.valid)
  val pastReadyX = past(dut.io.x.stream.ready)
  val pastPayloadX = past(dut.io.x.stream.payload)
  when(pastValidX && !pastReadyX) {
    assume(dut.io.x.stream.valid)
    assume(dut.io.x.stream.payload === pastPayloadX)
  }

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val xVal = dut.io.x.stream.payload(i)
    expectedPayload(i) := Mux(xVal.sign, spinalML.utils.Float.zero(2, 1), xVal)
  }

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  
  when(fireOut) {
    for(i <- 0 until 2) {
      assert(dut.io.y.stream.payload(i).asBits === expectedPayload(i).asBits, s"ReLU FP4 mismatch on lane $i")
    }
  }
}

object ReLUFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReLUFormal_I8, "relu_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReLUFormal_FP4, "relu_fp4")
  }
}
