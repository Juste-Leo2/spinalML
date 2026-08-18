package spinalML.symbolicTest.layers

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.tensors.Tensor
import spinalML.layers.LayerNorm1D

case class LayerNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val channels = 2
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(1, channels), lanes = channels))
    val gamma = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val beta = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(1, channels), lanes = channels))
  }
  val norm = LayerNorm1D(dataType, channels, 1)
  norm.io.x <> io.x
  norm.io.gamma <> io.gamma
  norm.io.beta <> io.beta
  io.y <> norm.io.y
}

class LayerNormFormal_I8 extends Component {
  val dut = FormalDut(LayerNormTestComp(SInt(8 bits)))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.gamma.stream.valid)
  anyseq(dut.io.gamma.stream.payload)
  anyseq(dut.io.beta.stream.valid)
  anyseq(dut.io.beta.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.y.stream.ready)
  
  // We force gamma and beta to be valid so the FSM doesn't stall indefinitely
  assume(dut.io.gamma.stream.valid)
  assume(dut.io.beta.stream.valid)
  // X can be randomly valid/invalid
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

class LayerNormFormal_FP4 extends Component {
  val dut = FormalDut(LayerNormTestComp(FP4_E2M1()))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.gamma.stream.valid)
  anyseq(dut.io.gamma.stream.payload)
  anyseq(dut.io.beta.stream.valid)
  anyseq(dut.io.beta.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.y.stream.ready)
  
  assume(dut.io.gamma.stream.valid)
  assume(dut.io.beta.stream.valid)
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

object LayerNormFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new LayerNormFormal_I8, "layernorm_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(8)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new LayerNormFormal_FP4, "layernorm_fp4")
  }
}
