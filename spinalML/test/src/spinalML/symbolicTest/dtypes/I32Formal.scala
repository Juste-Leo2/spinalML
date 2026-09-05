// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.dtypes

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I32

case class I32MathComp() extends Component {
  val io = new Bundle {
    val a = in(I32())
    val b = in(I32())
    val c_add = out(I32())
    val c_sub = out(I32())
    val c_mul = out(I32())
  }

  io.c_add := (io.a + io.b).resize(32)
  io.c_sub := (io.a - io.b).resize(32)
  io.c_mul := (io.a * io.b).resize(32)
}

class I32Formal extends Component {
  val dut = FormalDut(I32MathComp())

  // Drive inputs with free variables
  anyseq(dut.io.a)
  anyseq(dut.io.b)

  assumeInitial(clockDomain.isResetActive)

  // Golden model in Scala/Spinal (bit-exact)
  val expected_add = (dut.io.a + dut.io.b).resize(32)
  val expected_sub = (dut.io.a - dut.io.b).resize(32)
  val expected_mul = (dut.io.a * dut.io.b).resize(32)

  // Assertions
  assert(dut.io.c_add === expected_add, "I32 add mismatch")
  assert(dut.io.c_sub === expected_sub, "I32 sub mismatch")
  assert(dut.io.c_mul === expected_mul, "I32 mul mismatch")
}

object I32Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new I32Formal, "i32_formal")
  }
}
