package spinalML.symbolicTest.dtypes

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.U4

case class U4MathComp() extends Component {
  val io = new Bundle {
    val a = in(U4())
    val b = in(U4())
    val c_add = out(U4())
    val c_sub = out(U4())
    val c_mul = out(U4())
  }

  io.c_add := (io.a + io.b).resize(4)
  io.c_sub := (io.a - io.b).resize(4)
  io.c_mul := (io.a * io.b).resize(4)
}

class U4Formal extends Component {
  val dut = FormalDut(U4MathComp())

  // Drive inputs with free variables
  anyseq(dut.io.a)
  anyseq(dut.io.b)

  assumeInitial(clockDomain.isResetActive)

  // Golden model in Scala/Spinal (bit-exact)
  val expected_add = (dut.io.a + dut.io.b).resize(4)
  val expected_sub = (dut.io.a - dut.io.b).resize(4)
  val expected_mul = (dut.io.a * dut.io.b).resize(4)

  // Assertions
  assert(dut.io.c_add === expected_add, "U4 add mismatch")
  assert(dut.io.c_sub === expected_sub, "U4 sub mismatch")
  assert(dut.io.c_mul === expected_mul, "U4 mul mismatch")
}

object U4Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(2)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new U4Formal, "u4_formal")
  }
}
