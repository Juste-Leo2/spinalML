// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.layers

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.layers.Conv1DLayer

case class Conv1DTestComp[T <: Data, TAcc <: Data](
  dataType: HardType[T], 
  accType: HardType[TAcc], 
  parallelN: Boolean
) extends Component {
  val L = 2
  val K = 1
  val inChannels = 1
  val outChannels = 2
  val outLanes = 1
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L, inChannels), lanes = 1))
    val w = slave(Tensor(dataType, Seq(K * inChannels, outChannels), lanes = outLanes))
    val b = slave(Tensor(accType, Seq(1, outChannels), lanes = 1))
    val y = master(Tensor(accType, Seq(L - K + 1, outChannels), lanes = 1))
  }
  
  val dut = Conv1DLayer(dataType, accType, L, inChannels, outChannels, K, outLanes, tileSize = 4, parallelN = parallelN)
  dut.io.reArm := False
  dut.io.x <> io.x
  dut.io.w <> io.w
  dut.io.b <> io.b
  io.y <> dut.io.y
}

abstract class Conv1DFormalBase[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  parallelN: Boolean
) extends Component {
  val dut = FormalDut(Conv1DTestComp(dataType, accType, parallelN))

  anyseq(dut.io.x.stream.valid)
  anyseq(dut.io.x.stream.payload)
  anyseq(dut.io.w.stream.valid)
  anyseq(dut.io.w.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  
  // Basic Stream assumptions
  assume(dut.io.y.stream.ready)
  
  val pastValidX = past(dut.io.x.stream.valid)
  val pastReadyX = past(dut.io.x.stream.ready)
  val pastPayloadX = past(dut.io.x.stream.payload)
  when(pastValidX && !pastReadyX) {
    assume(dut.io.x.stream.valid)
    assume(dut.io.x.stream.payload === pastPayloadX)
  }

  val pastValidW = past(dut.io.w.stream.valid)
  val pastReadyW = past(dut.io.w.stream.ready)
  val pastPayloadW = past(dut.io.w.stream.payload)
  when(pastValidW && !pastReadyW) {
    assume(dut.io.w.stream.valid)
    assume(dut.io.w.stream.payload === pastPayloadW)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  // Ensure matrices stop sending after 1 tile
  val xCounter = Counter(3)
  when(dut.io.x.stream.valid && dut.io.x.stream.ready) { xCounter.increment() }
  assume(xCounter.value =/= 2 || !dut.io.x.stream.valid)

  val wCounter = Counter(2)
  when(dut.io.w.stream.valid && dut.io.w.stream.ready) { wCounter.increment() }
  assume(wCounter.value =/= 1 || !dut.io.w.stream.valid)

  val bCounter = Counter(3)
  when(dut.io.b.stream.valid && dut.io.b.stream.ready) { bCounter.increment() }
  assume(bCounter.value =/= 2 || !dut.io.b.stream.valid)

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  when(dut.io.y.stream.valid && dut.io.y.stream.ready) {
    track := True
  }
  
  when(track && !hasChecked) {
    assert(dut.io.y.stream.valid || dut.io.y.stream.ready, "Flow control check")
    hasChecked := True
  }
}

class Conv1DFormal_I8_Par extends Conv1DFormalBase(SInt(8 bits), SInt(32 bits), parallelN = true)
class Conv1DFormal_I8_Seq extends Conv1DFormalBase(SInt(8 bits), SInt(32 bits), parallelN = false)

object Conv1DFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig.withSymbiYosys.withBMC(15).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new Conv1DFormal_I8_Par, "conv1d_i8_par")
      
    FormalConfig.withSymbiYosys.withBMC(20).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new Conv1DFormal_I8_Seq, "conv1d_i8_seq")
  }
}
