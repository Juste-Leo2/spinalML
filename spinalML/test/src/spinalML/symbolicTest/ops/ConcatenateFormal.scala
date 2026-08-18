package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import spinalML.ops.ConcatenateTestComp

case class ConcatenateAxis1TestCompFormal[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(4, 4), lanes = 4))
  }
  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 1)
}

class ConcatenateAxis0Formal extends Component {
  val dut = FormalDut(new ConcatenateTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  
  // dut.io.a and b are stable through their reading phases? 
  // No, anyseq changes every cycle!
  // Wait, if anyseq changes every cycle, the payload we compare when c is valid 
  // MUST be the payload that was on a (or b) at that EXACT cycle.
  // In ConcatenateAxis0Op, `io.c.stream.payload := io.a.stream.payload` directly without delay!
  // So the combinatorial assertion works! c.payload is always a.payload or b.payload in the current cycle.
  
  val cCounter = Counter(6)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    when(cCounter.value < 2) {
      for(lane <- 0 until 2) assert(dut.io.c.stream.payload(lane) === dut.io.a.stream.payload(lane), "Concat A mismatch")
    } otherwise {
      for(lane <- 0 until 2) assert(dut.io.c.stream.payload(lane) === dut.io.b.stream.payload(lane), "Concat B mismatch")
    }
    cCounter.increment()
  }
  
  cover(cCounter.value === 5 && dut.io.c.stream.valid && dut.io.c.stream.ready)
}

class ConcatenateAxis1Formal extends Component {
  val dut = FormalDut(new ConcatenateAxis1TestCompFormal(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    for(lane <- 0 until 2) {
      assert(dut.io.c.stream.payload(lane) === dut.io.a.stream.payload(lane))
      assert(dut.io.c.stream.payload(2 + lane) === dut.io.b.stream.payload(lane))
    }
  }
}

object ConcatenateFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ConcatenateAxis0Formal, "concat_axis0_i8")

    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ConcatenateAxis1Formal, "concat_axis1_i8")
  }
}
