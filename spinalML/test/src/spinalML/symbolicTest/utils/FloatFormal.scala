package spinalML.symbolicTest.utils

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.FP8_E4M3
import spinalML.dtypes.FloatML
import spinalML.utils.Float

case class FloatUtilsTestComp() extends Component {
  val io = new Bundle {
    val a_sint = in(SInt(8 bits))
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
