// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}
import spinalML.ops.SubTestComp

class SubFormal_I8 extends Component {
  val dut = FormalDut(SubTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(I8(), 2)
  for(i <- 0 until 2) {
    expectedPayload(i).assignFrom((dut.io.a.stream.payload(i) - dut.io.b.stream.payload(i)).resized)
  }

  val trackedExpected = Reg(Vec(I8(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"Sub I8 mismatch on lane $i")
    }
    hasChecked := True
  }
}

class SubFormal_FP4 extends Component {
  val dut = FormalDut(SubTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val b = dut.io.b.stream.payload(i).asInstanceOf[FloatML]
    val invertedB = FloatML(b.expBits, b.mantBits)
    invertedB.exponent := b.exponent
    invertedB.mantissa := b.mantissa
    invertedB.sign := !b.sign
    expectedPayload(i).assignFrom(spinalML.utils.Float.add(dut.io.a.stream.payload(i), invertedB).asInstanceOf[FloatML])
  }

  val trackedExpected = Reg(Vec(FP4_E2M1(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"Sub FP4 mismatch on lane $i")
    }
    hasChecked := True
  }
}

object SubFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SubFormal_I8, "sub_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SubFormal_FP4, "sub_fp4")
  }
}
