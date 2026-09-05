// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.activations

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.tensors.Tensor
import spinalML.activations.LeakyReLUOp

case class LeakyReLUTestComp[T <: Data](dataType: HardType[T], shift: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(2), lanes = 2))
    val y = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  val lrelu = LeakyReLUOp(dataType, Seq(2), 2, shift)
  lrelu.io.x <> io.x
  io.y <> lrelu.io.y
}

class LeakyReLUFormal_I8 extends Component {
  val dut = FormalDut(LeakyReLUTestComp(SInt(8 bits), shift = 2))

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
    expectedPayload(i) := Mux(xVal < 0, xVal >> 2, xVal)
  }

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  
  when(fireOut) {
    for(i <- 0 until 2) {
      assert(dut.io.y.stream.payload(i) === expectedPayload(i), s"LeakyReLU I8 mismatch on lane $i")
    }
  }
}

class LeakyReLUFormal_FP4 extends Component {
  val shiftVal = 1
  val dut = FormalDut(LeakyReLUTestComp(FP4_E2M1(), shift = shiftVal))

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
    
    val neg = FP4_E2M1()
    neg.sign := True
    neg.mantissa := xVal.mantissa
    val shiftedExp = xVal.exponent.intoSInt - shiftVal
    when(shiftedExp <= 0 || xVal.exponent === 0) {
      neg.exponent := 0
      neg.mantissa := 0
      neg.sign := False
    } otherwise {
      neg.exponent := shiftedExp.asUInt.resized
    }
    
    expectedPayload(i) := Mux(xVal.sign, neg, xVal)
  }

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  
  when(fireOut) {
    for(i <- 0 until 2) {
      assert(dut.io.y.stream.payload(i).asBits === expectedPayload(i).asBits, s"LeakyReLU FP4 mismatch on lane $i")
    }
  }
}

object LeakyReLUFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new LeakyReLUFormal_I8, "leakyrelu_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new LeakyReLUFormal_FP4, "leakyrelu_fp4")
  }
}
