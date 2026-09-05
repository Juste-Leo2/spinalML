// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FloatML}
import spinalML.ops.CastOp
import spinalML.utils.Float

class CastFormal extends Component {
  val dut = FormalDut(CastOp(
    dataTypeIn = I8(),
    dataTypeOut = FloatML(4, 3),
    shape = Seq(1),
    lanes = 1
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)
  
  // Golden model (Bit-exact with RTL)
  val valIn = dut.io.a.stream.payload(0)
  val expected = Float.fromSInt(valIn, 4, 3)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected, "cast data mismatch")
  }
}

object CastFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new CastFormal, "cast_formal")
  }
}
