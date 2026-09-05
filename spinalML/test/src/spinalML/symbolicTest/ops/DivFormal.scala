// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}
import spinalML.ops.DivTestComp
import spinalML.utils.MathLUTs

class DivFormal_I8 extends Component {
  val dut = FormalDut(DivTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  
  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }
  
  // Assume b is not 0 for division
  for (i <- 0 until 2) {
    assume(dut.io.b.stream.payload(i).asBits.asUInt =/= 0)
  }

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.intValFn(8)
  val encodeFn = MathLUTs.intEncodeFn(8)
  val romContent = for(i <- 0 until 256) yield {
    val resDouble = mathFn(valFn(i))
    U(encodeFn(resDouble), 8 bits)
  }
  val goldenRecipRom = Mem(UInt(8 bits), initialContent = romContent)

  val trackedExpected = Reg(Vec(I8(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)
  val fireIn = dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    val expectedPayload = Vec(I8(), 2)
    for(i <- 0 until 2) {
      val bBits = dut.io.b.stream.payload(i).asBits.asUInt
      val goldenRecipBits = goldenRecipRom.readAsync(bBits)
      val goldenRecipSInt = goldenRecipBits.asSInt
      expectedPayload(i).assignFrom((dut.io.a.stream.payload(i).asInstanceOf[SInt] * goldenRecipSInt).resized)
    }
    trackedExpected := expectedPayload
  }
  
  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"Div I8 mismatch on lane $i")
    }
    hasChecked := True
  }
}

class DivFormal_FP4 extends Component {
  val dut = FormalDut(DivTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  
  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  // Assume b is not 0 for division
  for (i <- 0 until 2) {
    assume(dut.io.b.stream.payload(i).asBits.asUInt =/= 0)
  }

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.floatValFn(2, 1)
  val encodeFn = MathLUTs.floatEncodeFn(2, 1)
  val romContent = for(i <- 0 until 16) yield {
    val resDouble = mathFn(valFn(i))
    U(encodeFn(resDouble), 4 bits)
  }
  val goldenRecipRom = Mem(UInt(4 bits), initialContent = romContent)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val bBits = dut.io.b.stream.payload(i).asBits.asUInt
    val goldenRecipBits = goldenRecipRom.readAsync(bBits)
    
    val goldenRecipFloat = FP4_E2M1()
    goldenRecipFloat.assignFromBits(goldenRecipBits.asBits)
    
    expectedPayload(i).assignFrom(spinalML.utils.Float.mul(dut.io.a.stream.payload(i), goldenRecipFloat).asInstanceOf[FloatML])
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
      assert(dut.io.c.stream.payload(i).asBits === trackedExpected(i).asBits, s"Div FP4 mismatch on lane $i")
    }
    hasChecked := True
  }
}

object DivFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DivFormal_I8, "div_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DivFormal_FP4, "div_fp4")
  }
}
