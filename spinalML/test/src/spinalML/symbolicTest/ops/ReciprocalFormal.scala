package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.ops.ReciprocalTestComp
import spinalML.utils.MathLUTs

class ReciprocalFormal_FP4 extends Component {
  val dut = FormalDut(ReciprocalTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.floatValFn(2, 1)
  val encodeFn = MathLUTs.floatEncodeFn(2, 1)
  val romContent = for(i <- 0 until 16) yield {
    val resDouble = mathFn(valFn(i))
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

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i).asBits === trackedExpected(i).asBits, s"Reciprocal FP4 mismatch on lane $i")
    }
    track := False
  }
}

class ReciprocalFormal_FP9 extends Component {
  val dut = FormalDut(ReciprocalTestComp(FloatML(4, 4)))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.floatValFn(4, 4)
  val encodeFn = MathLUTs.floatEncodeFn(4, 4)
  val romContent = for(i <- 0 until 512) yield {
    val resDouble = mathFn(valFn(i))
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
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.valid, "Flow control drop")
    }
    hasChecked := True
  }
}

class ReciprocalFormal_FP4_Alg extends Component {
  val dut = FormalDut(ReciprocalTestComp(FP4_E2M1(), forceAlg = true))

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

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.floatValFn(2, 1)
  val encodeFn = MathLUTs.floatEncodeFn(2, 1)
  val romContent = for(i <- 0 until 16) yield {
    val resDouble = mathFn(valFn(i))
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
      assert(dut.io.c.stream.valid, "Flow control drop")
    }
    hasChecked := True
  }
}

class ReciprocalFormal_I10 extends Component {
  val dut = FormalDut(ReciprocalTestComp(SInt(10 bits)))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val track = RegInit(False)
  val hasChecked = RegInit(False)
  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }
  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.c.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

object ReciprocalFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReciprocalFormal_FP4, "reciprocal_fp4")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReciprocalFormal_FP4_Alg, "reciprocal_fp4_alg")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReciprocalFormal_FP9, "reciprocal_fp9")
      
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReciprocalFormal_I10, "reciprocal_i10")
  }
}
