package spinalML.symbolicTest.dtypes

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.U8

case class U8MathComp() extends Component {
  val io = new Bundle {
    val a = in(U8())
    val b = in(U8())
    val c_add = out(U8())
    val c_sub = out(U8())
    val c_mul = out(U8())
  }

  io.c_add := (io.a + io.b).resize(8)
  io.c_sub := (io.a - io.b).resize(8)
  io.c_mul := (io.a * io.b).resize(8)
}

class U8Formal extends Component {
  val dut = FormalDut(U8MathComp())

  // Drive inputs with free variables
  anyseq(dut.io.a)
  anyseq(dut.io.b)

  assumeInitial(clockDomain.isResetActive)

  // Golden model in Scala/Spinal (bit-exact)
  val expected_add = (dut.io.a + dut.io.b).resize(8)
  val expected_sub = (dut.io.a - dut.io.b).resize(8)
  val expected_mul = (dut.io.a * dut.io.b).resize(8)

  // Assertions
  assert(dut.io.c_add === expected_add, "U8 add mismatch")
  assert(dut.io.c_sub === expected_sub, "U8 sub mismatch")
  assert(dut.io.c_mul === expected_mul, "U8 mul mismatch")
}

object U8Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new U8Formal, "u8_formal")
  }
}
