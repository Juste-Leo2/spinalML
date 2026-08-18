package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, I32}
import spinalML.ops.RequantizeOp

class requantizeFormal extends Component {
  val shift = 4
  val dut = FormalDut(RequantizeOp(
    dataTypeIn = I32(),
    dataTypeOut = I8(),
    shape = Seq(1),
    lanes = 1,
    shift = shift
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Pure combinational op, no latency introduced here.
  // Wait! RequantizeOp uses arbitrationFrom, which means it is 0 latency.
  
  // Golden model (Bit-exact with RTL)
  val valIn = dut.io.a.stream.payload(0)
  val shifted = (valIn >> shift).resize(32 bits)
  
  val maxVal = (1 << 7) - 1
  val minVal = -(1 << 7)
  
  val saturated = Mux(shifted > maxVal, S(maxVal, 32 bits),
                    Mux(shifted < minVal, S(minVal, 32 bits),
                        shifted))
  val expected = saturated.resize(8 bits)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected, "requantize data mismatch")
  }
}

object requantizeFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new requantizeFormal, "requantize_formal")
  }
}
