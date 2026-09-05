// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}
import spinalML.ops.ScaleAddTestComp

class ScaleAddFormal_I8 extends Component {
  val dut = FormalDut(ScaleAddTestComp(I8()))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.x.stream.valid)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(I8(), 2)
  for(i <- 0 until 2) {
    val x = dut.io.x.stream.payload(i).asInstanceOf[SInt]
    val a = dut.io.a.stream.payload(i).asInstanceOf[SInt]
    val b = dut.io.b.stream.payload(i).asInstanceOf[SInt]
    expectedPayload(i).assignFrom(((x * a) + b).resized)
  }

  val trackedExpected = Reg(Vec(I8(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"ScaleAdd I8 mismatch on lane $i")
    }
    hasChecked := True
  }
}

class ScaleAddFormal_FP4 extends Component {
  val dut = FormalDut(ScaleAddTestComp(FP4_E2M1()))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.x.stream.valid)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val x = dut.io.x.stream.payload(i).asInstanceOf[FloatML]
    val a = dut.io.a.stream.payload(i).asInstanceOf[FloatML]
    val b = dut.io.b.stream.payload(i).asInstanceOf[FloatML]
    val mulRes = spinalML.utils.Float.mul(x, a)
    val addRes = spinalML.utils.Float.add(mulRes, b)
    expectedPayload(i).assignFrom(addRes.asInstanceOf[FloatML])
  }

  val trackedExpected = Reg(Vec(FP4_E2M1(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"ScaleAdd FP4 mismatch on lane $i")
    }
    hasChecked := True
  }
}

object ScaleAddFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ScaleAddFormal_I8, "scaleadd_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ScaleAddFormal_FP4, "scaleadd_fp4")
  }
}
