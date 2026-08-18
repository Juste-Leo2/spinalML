package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.ops.Seq2ColTestComp_3_K2

class Seq2ColFormal extends Component {
  val dut = FormalDut(new Seq2ColTestComp_3_K2(I8()))

  val goldenInput = Vec.fill(3)(I8())
  anyconst(goldenInput)

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Track writes
  val writeCount = Counter(4)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) {
    when(writeCount.value < 3) {
      writeCount.increment()
    }
  }
  
  when(writeCount.value < 3) {
    assume(dut.io.a.stream.payload(0) === goldenInput(writeCount.value))
  }

  val readCount = Counter(3)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    when(readCount.value === 0) {
      assert(dut.io.c.stream.payload(0) === goldenInput(0))
      assert(dut.io.c.stream.payload(1) === goldenInput(1))
    } elsewhen(readCount.value === 1) {
      assert(dut.io.c.stream.payload(0) === goldenInput(1))
      assert(dut.io.c.stream.payload(1) === goldenInput(2))
    }
    
    when(readCount.value < 2) {
      readCount.increment()
    }
  }
  
  cover(readCount.value === 2)
}

object Seq2ColFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15) 
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new Seq2ColFormal, "seq2col_3_k2_i8")
  }
}
