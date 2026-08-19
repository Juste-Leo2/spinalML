package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.ops.RepackTestComp

class RepackFormal extends Component {
  val dut = FormalDut(new RepackTestComp(I8()))

  val goldenInput = Vec.fill(4)(I8())
  anyconst(goldenInput)

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Track writes
  val writeCount = Counter(3) // 0 to 2
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) {
    when(writeCount.value === 0) {
      writeCount.increment()
    } elsewhen(writeCount.value === 1) {
      writeCount.increment()
    }
  }
  
  when(writeCount.value === 0) {
    assume(dut.io.a.stream.payload(0) === goldenInput(0))
    assume(dut.io.a.stream.payload(1) === goldenInput(1))
  } elsewhen(writeCount.value === 1) {
    assume(dut.io.a.stream.payload(0) === goldenInput(2))
    assume(dut.io.a.stream.payload(1) === goldenInput(3))
  }

  val readCount = Counter(2)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    when(readCount.value === 0) {
      assert(dut.io.c.stream.payload(0) === goldenInput(0))
      assert(dut.io.c.stream.payload(1) === goldenInput(1))
      assert(dut.io.c.stream.payload(2) === goldenInput(2))
      assert(dut.io.c.stream.payload(3) === goldenInput(3))
      readCount.increment()
    }
  }
  
  cover(readCount.value === 1)
}

object RepackFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4) 
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new RepackFormal, "repack_i8_2_to_4")
  }
}
