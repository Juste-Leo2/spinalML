package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}
import spinalML.ops.AbsTestComp

class AbsFormal_I8 extends Component {
  val dut = FormalDut(AbsTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(I8(), 2)
  for(i <- 0 until 2) {
    val a = dut.io.a.stream.payload(i).asInstanceOf[SInt]
    expectedPayload(i).assignFrom(Mux(a < 0, -a, a).asInstanceOf[SInt])
  }

  val trackedExpected = Reg(Vec(I8(), 2))
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
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"Abs I8 mismatch on lane $i")
    }
    hasChecked := True
  }
}

class AbsFormal_FP4 extends Component {
  val dut = FormalDut(AbsTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val a = dut.io.a.stream.payload(i).asInstanceOf[FloatML]
    val outF = FloatML(a.expBits, a.mantBits)
    outF.sign := False
    outF.exponent := a.exponent
    outF.mantissa := a.mantissa
    expectedPayload(i).assignFrom(outF)
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
      assert(dut.io.c.stream.payload(i) === trackedExpected(i), s"Abs FP4 mismatch on lane $i")
    }
    hasChecked := True
  }
}

object AbsFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AbsFormal_I8, "abs_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AbsFormal_FP4, "abs_fp4")
  }
}
