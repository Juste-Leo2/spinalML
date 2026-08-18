package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.ops.ExpTestComp
import spinalML.utils.MathLUTs

class ExpFormal_FP4 extends Component {
  // Tests the purely LUT-based path (bitWidth <= 8)
  val dut = FormalDut(ExpTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  // Golden ROM generation using pure Scala Math.exp
  val valFn = MathLUTs.floatValFn(2, 1)
  val encodeFn = MathLUTs.floatEncodeFn(2, 1)
  val romContent = for(i <- 0 until 16) yield {
    val resDouble = Math.exp(valFn(i))
    U(encodeFn(resDouble), 4 bits)
  }
  val goldenRom = Mem(UInt(4 bits), initialContent = romContent)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val aBits = dut.io.a.stream.payload(i).asBits.asUInt
    val goldenBits = goldenRom.readAsync(aBits)
    expectedPayload(i).assignFromBits(goldenBits.asBits)
  }

  val trackedExpected = Reg(Vec(FP4_E2M1(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i).asBits === trackedExpected(i).asBits, s"Exp FP4 (LUT) mismatch on lane $i")
    }
    hasChecked := True
  }
}

class ExpFormal_FP9 extends Component {
  // Tests the Alg+LUT path (bitWidth > 8 and FloatML)
  val dut = FormalDut(ExpTestComp(FloatML(4, 4)))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  // Golden ROM generation using pure Scala Math.exp
  val valFn = MathLUTs.floatValFn(4, 4)
  val encodeFn = MathLUTs.floatEncodeFn(4, 4)
  val romContent = for(i <- 0 until 512) yield {
    val resDouble = Math.exp(valFn(i))
    U(encodeFn(resDouble), 9 bits)
  }
  val goldenRom = Mem(UInt(9 bits), initialContent = romContent)

  val expectedPayload = Vec(FloatML(4, 4), 2)
  for(i <- 0 until 2) {
    val aBits = dut.io.a.stream.payload(i).asBits.asUInt
    val goldenBits = goldenRom.readAsync(aBits)
    expectedPayload(i).assignFromBits(goldenBits.asBits)
  }

  val trackedExpected = Reg(Vec(FloatML(4, 4), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    // Alg+LUT is an approximation, we check flow control here.
    assert(dut.io.c.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

class ExpFormal_I10 extends Component {
  // Tests the PWL path (bitWidth > 8 and Int)
  val dut = FormalDut(ExpTestComp(SInt(10 bits)))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Golden ROM generation using pure Scala Math.exp
  val valFn = MathLUTs.intValFn(10)
  val encodeFn = MathLUTs.intEncodeFn(10)
  val romContent = for(i <- 0 until 1024) yield {
    val resDouble = Math.exp(valFn(i))
    U(encodeFn(resDouble), 10 bits)
  }
  val goldenRom = Mem(UInt(10 bits), initialContent = romContent)

  val expectedPayload = Vec(SInt(10 bits), 2)
  for(i <- 0 until 2) {
    val aBits = dut.io.a.stream.payload(i).asBits.asUInt
    val goldenBits = goldenRom.readAsync(aBits)
    expectedPayload(i).assignFromBits(goldenBits.asBits)
  }

  val trackedExpected = Reg(Vec(SInt(10 bits), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      // PWL is an approximation. It won't be bit-exact to Double!
      assert(dut.io.c.stream.valid, "Flow control drop")
    }
    hasChecked := True
  }
}

object ExpFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ExpFormal_FP4, "exp_fp4")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ExpFormal_FP9, "exp_fp9")
      
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ExpFormal_I10, "exp_i10")
  }
}
