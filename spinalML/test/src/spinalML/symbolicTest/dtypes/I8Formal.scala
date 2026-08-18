package spinalML.symbolicTest.dtypes

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8

case class I8MathComp() extends Component {
  val io = new Bundle {
    val a = in(I8())
    val b = in(I8())
    val c_add = out(I8())
    val c_sub = out(I8())
    val c_mul = out(I8())
  }

  io.c_add := (io.a + io.b).resize(8)
  io.c_sub := (io.a - io.b).resize(8)
  io.c_mul := (io.a * io.b).resize(8)
}

class I8Formal extends Component {
  val dut = FormalDut(I8MathComp())

  // Drive inputs with free variables
  anyseq(dut.io.a)
  anyseq(dut.io.b)

  assumeInitial(clockDomain.isResetActive)

  // Golden model in Scala/Spinal (bit-exact)
  val expected_add = (dut.io.a + dut.io.b).resize(8)
  val expected_sub = (dut.io.a - dut.io.b).resize(8)
  val expected_mul = (dut.io.a * dut.io.b).resize(8)

  // Assertions
  assert(dut.io.c_add === expected_add, "I8 add mismatch")
  assert(dut.io.c_sub === expected_sub, "I8 sub mismatch")
  assert(dut.io.c_mul === expected_mul, "I8 mul mismatch")
}

object I8Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new I8Formal, "i8_formal")
  }
}
