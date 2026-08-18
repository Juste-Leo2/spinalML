package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import spinalML.ops.TransposeTestComp_2x3

class TransposeFormal extends Component {
  val dut = FormalDut(new TransposeTestComp_2x3(I8()))

  val goldenInput = Vec.fill(6)(I8())
  anyconst(goldenInput)

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Track writes
  val writeCount = Counter(7) // 0 to 6
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) {
    when(writeCount.value < 6) {
      writeCount.increment()
    }
  }
  
  // Constrain inputs to match our golden matrix
  when(writeCount.value < 6) {
    assume(dut.io.a.stream.payload(0) === goldenInput(writeCount.value))
  }

  // Expected transposed indices: 
  // Original (2x3): [0, 1, 2; 3, 4, 5]
  // Transposed (3x2): [0, 3; 1, 4; 2, 5]
  val expectedIndices = Vec(U(0, 3 bits), U(3, 3 bits), U(1, 3 bits), U(4, 3 bits), U(2, 3 bits), U(5, 3 bits))
  
  val readCount = Counter(7) // 0 to 6
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    when(readCount.value < 6) {
      assert(dut.io.c.stream.payload(0) === goldenInput(expectedIndices(readCount.value)), "Transpose mismatch")
      readCount.increment()
    }
  }
  
  cover(readCount.value === 6)
}

object TransposeFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15) // Enough for 6 writes + 1 flush + 6 reads = 13 cycles min
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TransposeFormal, "transpose_2x3_i8")
  }
}
