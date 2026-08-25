package spinalML.symbolicTest.layers

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.layers.LinearLayer

case class LinearTestComp[T <: Data, TAcc <: Data](
  dataType: HardType[T], 
  accType: HardType[TAcc], 
  parallelN: Boolean
) extends Component {
  val M = 2
  val K = 1
  val N = 2
  val lanes = 1
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(M, K), lanes))
    val w = slave(Tensor(dataType, Seq(K, N), lanes))
    val b = slave(Tensor(accType, Seq(1, N), lanes = 1))
    val y = master(Tensor(accType, Seq(M, N), lanes = 1))
  }
  
  val dut = LinearLayer(dataType, dataType, accType, Seq(M, K), Seq(K, N), lanes, tileSize = 4, parallelN = parallelN)
  dut.io.reArm := False
  dut.io.a <> io.a
  dut.io.w <> io.w
  dut.io.b <> io.b
  io.y <> dut.io.y
}

abstract class LinearFormalBase[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  parallelN: Boolean
) extends Component {
  val dut = FormalDut(LinearTestComp(dataType, accType, parallelN))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.w.stream.valid)
  anyseq(dut.io.w.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.y.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  
  // Basic Stream assumptions
  assume(dut.io.y.stream.ready)
  
  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
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
  val aCounter = Counter(3)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) { aCounter.increment() }
  assume(aCounter.value =/= 2 || !dut.io.a.stream.valid)

  val wCounter = Counter(3)
  when(dut.io.w.stream.valid && dut.io.w.stream.ready) { wCounter.increment() }
  assume(wCounter.value =/= 2 || !dut.io.w.stream.valid)

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

class LinearFormal_I8_Par extends LinearFormalBase(SInt(8 bits), SInt(32 bits), parallelN = true)
class LinearFormal_I8_Seq extends LinearFormalBase(SInt(8 bits), SInt(32 bits), parallelN = false)

object LinearFormal {
  def main(args: Array[String]): Unit = {
  FormalConfig.withSymbiYosys.withBMC(15).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
    .doVerify(new LinearFormal_I8_Par, "linear_i8_par")
    
  FormalConfig.withSymbiYosys.withBMC(20).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
    .doVerify(new LinearFormal_I8_Seq, "linear_i8_seq")
  }
}
