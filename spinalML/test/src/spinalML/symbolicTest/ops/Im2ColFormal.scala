package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.ops.Im2ColTestComp_3x3_K2

class Im2ColFormal extends Component {
  val dut = FormalDut(new Im2ColTestComp_3x3_K2(I8()))

  val goldenInput = Vec.fill(9)(I8())
  anyconst(goldenInput)

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Track writes
  val writeCount = Counter(10)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) {
    when(writeCount.value < 9) {
      writeCount.increment()
    }
  }
  
  when(writeCount.value < 9) {
    assume(dut.io.a.stream.payload(0) === goldenInput(writeCount.value))
  }

  // Windows expectations
  // Image is 3x3:
  // 0 1 2
  // 3 4 5
  // 6 7 8
  // Window 0 (0,0): 0, 1, 3, 4
  // Window 1 (0,1): 1, 2, 4, 5
  // Window 2 (1,0): 3, 4, 6, 7
  // Window 3 (1,1): 4, 5, 7, 8
  
  val readCount = Counter(5)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    when(readCount.value === 0) {
      assert(dut.io.c.stream.payload(0) === goldenInput(0))
      assert(dut.io.c.stream.payload(1) === goldenInput(1))
      assert(dut.io.c.stream.payload(2) === goldenInput(3))
      assert(dut.io.c.stream.payload(3) === goldenInput(4))
    } elsewhen(readCount.value === 1) {
      assert(dut.io.c.stream.payload(0) === goldenInput(1))
      assert(dut.io.c.stream.payload(1) === goldenInput(2))
      assert(dut.io.c.stream.payload(2) === goldenInput(4))
      assert(dut.io.c.stream.payload(3) === goldenInput(5))
    } elsewhen(readCount.value === 2) {
      assert(dut.io.c.stream.payload(0) === goldenInput(3))
      assert(dut.io.c.stream.payload(1) === goldenInput(4))
      assert(dut.io.c.stream.payload(2) === goldenInput(6))
      assert(dut.io.c.stream.payload(3) === goldenInput(7))
    } elsewhen(readCount.value === 3) {
      assert(dut.io.c.stream.payload(0) === goldenInput(4))
      assert(dut.io.c.stream.payload(1) === goldenInput(5))
      assert(dut.io.c.stream.payload(2) === goldenInput(7))
      assert(dut.io.c.stream.payload(3) === goldenInput(8))
    }
    
    when(readCount.value < 4) {
      readCount.increment()
    }
  }
  
  cover(readCount.value === 4)
}

object Im2ColFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4) 
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new Im2ColFormal, "im2col_3x3_k2_i8")
  }
}
