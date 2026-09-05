// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.dtypes

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I16

case class I16MathComp() extends Component {
  val io = new Bundle {
    val a = in(I16())
    val b = in(I16())
    val c_add = out(I16())
    val c_sub = out(I16())
    val c_mul = out(I16())
  }

  io.c_add := (io.a + io.b).resize(16)
  io.c_sub := (io.a - io.b).resize(16)
  io.c_mul := (io.a * io.b).resize(16)
}

class I16Formal extends Component {
  val dut = FormalDut(I16MathComp())

  // Drive inputs with free variables
  anyseq(dut.io.a)
  anyseq(dut.io.b)

  assumeInitial(clockDomain.isResetActive)

  // Golden model in Scala/Spinal (bit-exact)
  val expected_add = (dut.io.a + dut.io.b).resize(16)
  val expected_sub = (dut.io.a - dut.io.b).resize(16)
  val expected_mul = (dut.io.a * dut.io.b).resize(16)

  // Assertions
  assert(dut.io.c_add === expected_add, "I16 add mismatch")
  assert(dut.io.c_sub === expected_sub, "I16 sub mismatch")
  assert(dut.io.c_mul === expected_mul, "I16 mul mismatch")
}

object I16Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new I16Formal, "i16_formal")
  }
}
