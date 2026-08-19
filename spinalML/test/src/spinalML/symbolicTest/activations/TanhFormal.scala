package spinalML.symbolicTest.activations

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML, I8}
import spinalML.activations.TanhTestComp

// Tanh is a composition of validated primitives (mul -> sigmoid -> 2*sig - 1).
// We only check flow control (no data drop) like SoftmaxFormal.

class TanhFormal_I8 extends Component {
  val dut = FormalDut(TanhTestComp(I8()))

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

class TanhFormal_FP4 extends Component {
  val dut = FormalDut(TanhTestComp(FP4_E2M1()))

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

class TanhFormal_FP9 extends Component {
  val dut = FormalDut(TanhTestComp(FloatML(4, 4)))

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

class TanhFormal_I10 extends Component {
  val dut = FormalDut(TanhTestComp(SInt(10 bits)))

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

object TanhFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TanhFormal_I8, "tanh_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TanhFormal_FP4, "tanh_fp4")

    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TanhFormal_FP9, "tanh_fp9")

    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TanhFormal_I10, "tanh_i10")
  }
}