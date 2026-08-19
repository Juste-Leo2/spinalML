package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1, FloatML}
import spinalML.tensors.Tensor
import spinalML.ops.MatmulOp

case class MatmulTestComp[T <: Data, TAcc <: Data](
  dataType: HardType[T], 
  accType: HardType[TAcc], 
  parallelN: Boolean
) extends Component {
  val M = 2
  val K = 2
  val N = 2
  val lanes = 2
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(M, K), lanes))
    val b = slave(Tensor(dataType, Seq(K, N), lanes))
    val c = master(Tensor(accType, Seq(M, N), lanes = 1))
  }
  
  val dut = MatmulOp(dataType, accType, Seq(M, K), Seq(K, N), lanes, parallelN = parallelN, pipelineTree = true)
  dut.io.a <> io.a
  dut.io.b <> io.b
  io.c <> dut.io.c
}

abstract class MatMulFormalBase[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  parallelN: Boolean
) extends Component {
  val dut = FormalDut(MatmulTestComp(dataType, accType, parallelN))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  
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

  // Ensure matrices stop sending after 1 tile
  val aCounter = Counter(3)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) { aCounter.increment() }
  assume(aCounter.value =/= 2 || !dut.io.a.stream.valid)

  val bCounter = Counter(3)
  when(dut.io.b.stream.valid && dut.io.b.stream.ready) { bCounter.increment() }
  assume(bCounter.value =/= 2 || !dut.io.b.stream.valid)
  
  // Trackers
  val aReg = Reg(Vec(Vec(dataType(), 2), 2))
  when(dut.io.a.stream.valid && dut.io.a.stream.ready && aCounter.value < 2) {
    aReg(aCounter.value.resized)(0) := dut.io.a.stream.payload(0)
    aReg(aCounter.value.resized)(1) := dut.io.a.stream.payload(1)
  }
  
  val bReg = Reg(Vec(Vec(dataType(), 2), 2))
  when(dut.io.b.stream.valid && dut.io.b.stream.ready && bCounter.value < 2) {
    bReg(bCounter.value.resized)(0) := dut.io.b.stream.payload(0)
    bReg(bCounter.value.resized)(1) := dut.io.b.stream.payload(1)
  }

  val cCounter = Counter(4)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    cCounter.increment()
  }

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    cCounter.increment()
    track := True
  }
  
  when(track && !hasChecked) {
    assert(dut.io.c.stream.valid || dut.io.c.stream.ready, "Flow control check")
    hasChecked := True
  }
}

// I8 (SInt) parallelN=true
class MatMulFormal_I8_Par extends MatMulFormalBase(SInt(8 bits), SInt(32 bits), parallelN = true)
// I8 (SInt) parallelN=false
class MatMulFormal_I8_Seq extends MatMulFormalBase(SInt(8 bits), SInt(32 bits), parallelN = false)
// FP4 parallelN=true
class MatMulFormal_FP4_Par extends MatMulFormalBase(FP4_E2M1(), FP4_E2M1(), parallelN = true)
// FP4 parallelN=false
class MatMulFormal_FP4_Seq extends MatMulFormalBase(FP4_E2M1(), FP4_E2M1(), parallelN = false)

object MatMulFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig.withSymbiYosys.withBMC(15).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new MatMulFormal_I8_Par, "matmul_i8_par")
      
    FormalConfig.withSymbiYosys.withBMC(12).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new MatMulFormal_I8_Seq, "matmul_i8_seq")
      
    FormalConfig.withSymbiYosys.withBMC(15).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new MatMulFormal_FP4_Par, "matmul_fp4_par")
      
    FormalConfig.withSymbiYosys.withBMC(12).withTimeout(600).withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))).workspacePath("formal")
      .doVerify(new MatMulFormal_FP4_Seq, "matmul_fp4_seq")
  }
}
