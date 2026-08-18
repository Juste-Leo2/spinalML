package spinalML.symbolicTest.activations

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.tensors.Tensor
import spinalML.activations.Softmax1D

case class SoftmaxTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val channels = 2
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(1, channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(1, channels), lanes = channels))
  }
  val softmax = Softmax1D(dataType, channels, 1)
  softmax.io.x <> io.x
  io.y <> softmax.io.y
}

class SoftmaxFormal_I8 extends Component {
  val dut = FormalDut(SoftmaxTestComp(SInt(8 bits)))

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

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.y.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

class SoftmaxFormal_FP4 extends Component {
  val dut = FormalDut(SoftmaxTestComp(FP4_E2M1()))

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

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.y.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

class SoftmaxFormal_FP9 extends Component {
  val dut = FormalDut(SoftmaxTestComp(FloatML(4, 4)))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.y.stream.ready)
  assume(dut.io.x.stream.valid)
  
  val pastValidX = past(dut.io.x.stream.valid)
  val pastReadyX = past(dut.io.x.stream.ready)
  val pastPayloadX = past(dut.io.x.stream.payload)
  when(pastValidX && !pastReadyX) {
    assume(dut.io.x.stream.valid)
    assume(dut.io.x.stream.payload === pastPayloadX)
  }

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.y.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

class SoftmaxFormal_I10 extends Component {
  val dut = FormalDut(SoftmaxTestComp(SInt(10 bits)))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.y.stream.ready)
  assume(dut.io.x.stream.valid)
  
  val pastValidX = past(dut.io.x.stream.valid)
  val pastReadyX = past(dut.io.x.stream.ready)
  val pastPayloadX = past(dut.io.x.stream.payload)
  when(pastValidX && !pastReadyX) {
    assume(dut.io.x.stream.valid)
    assume(dut.io.x.stream.payload === pastPayloadX)
  }

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.x.stream.valid && dut.io.x.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.y.stream.valid && dut.io.y.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.y.stream.valid, "Flow control drop")
    hasChecked := True
  }
}

object SoftmaxFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SoftmaxFormal_I8, "softmax_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SoftmaxFormal_FP4, "softmax_fp4")

    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SoftmaxFormal_FP9, "softmax_fp9")

    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SoftmaxFormal_I10, "softmax_i10")
  }
}
