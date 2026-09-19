// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.utils

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.FP8_E4M3
import spinalML.dtypes.FloatML
import spinalML.utils.Float

case class FloatUtilsTestComp() extends Component {
  val io = new Bundle {
    // 16-bit input so E4M3 saturation is reachable (|x| > 448 saturates).
    val a_sint = in(SInt(16 bits))
    val c_float = out(FloatML(4, 3))
    val c_zero = out(FloatML(4, 3))
  }

  io.c_float := Float.fromSInt(io.a_sint, 4, 3)
  io.c_zero := Float.zero(4, 3)
}

class FloatFormal extends Component {
  val dut = FormalDut(FloatUtilsTestComp())

  anyseq(dut.io.a_sint)

  assumeInitial(clockDomain.isResetActive)

  val expected_float = Float.fromSInt(dut.io.a_sint, 4, 3)
  val expected_zero = Float.zero(4, 3)

  assert(dut.io.c_float === expected_float, "Float.fromSInt mismatch")
  assert(dut.io.c_zero === expected_zero, "Float.zero mismatch")

  // DTYPE-07: E4M3 saturation never emits the NaN slot (exp 15, mant 7);
  // past-the-max saturates to exactly +/-448.
  assert(!(dut.io.c_float.exponent === U(15, 4 bits) && dut.io.c_float.mantissa === U(7, 3 bits)),
    "E4M3 emitted the NaN pattern (15, 7)")
  when(dut.io.a_sint > S(448, 16 bits)) {
    assert(!dut.io.c_float.sign, "E4M3 +sat sign")
    assert(dut.io.c_float.exponent === U(15, 4 bits), "E4M3 +sat exponent")
    assert(dut.io.c_float.mantissa === U(6, 3 bits), "E4M3 +sat mantissa (448)")
  }
  when(dut.io.a_sint < S(-448, 16 bits)) {
    assert(dut.io.c_float.sign, "E4M3 -sat sign")
    assert(dut.io.c_float.exponent === U(15, 4 bits), "E4M3 -sat exponent")
    assert(dut.io.c_float.mantissa === U(6, 3 bits), "E4M3 -sat mantissa (-448)")
  }
}

object FloatFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new FloatFormal, "float_formal")
  }
}
