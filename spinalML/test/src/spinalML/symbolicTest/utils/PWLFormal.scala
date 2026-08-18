package spinalML.symbolicTest.utils

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import spinalML.utils.UnaryPWLOp
import spinalML.utils.PWLLUTs

class PWLFormal extends Component {
  // A simple PWL operation mapping I8 to I8 with 1 segment (dummy math function)
  val segmentIndexFn = (x: SInt) => U(0, 1 bits)
  val segmentFn = (i: Int) => (1.0, 0.0) // y = 1*x + 0
  
  val dut = FormalDut(UnaryPWLOp(
    dataType = I8(),
    shape = Seq(1),
    lanes = 1,
    numSegments = 1,
    segmentIndexFn = segmentIndexFn,
    segmentFn = segmentFn
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)
  assume(dut.io.a.stream.payload === past(dut.io.a.stream.payload))

  // In this dummy PWL (y = x), the output should match the input, delayed by 1 cycle
  // UnaryPWLOp introduces exactly 1 cycle of latency (readSync)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === dut.io.a.stream.payload(0), "PWL data mismatch")
  }
}

object PWLFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(2)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new PWLFormal, "pwl_formal")
  }
}
