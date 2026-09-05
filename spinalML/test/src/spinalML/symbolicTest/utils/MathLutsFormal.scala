// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.utils

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.utils.UnaryLUTOp

class MathLutsFormal extends Component {
  val valFn = (i: Int) => i.toDouble
  val encodeFn = (y: Double) => BigInt(y.toInt)
  val mathFn = (x: Double) => x // y = x
  
  val dut = FormalDut(UnaryLUTOp(
    dataType = I8(),
    shape = Seq(1),
    lanes = 1,
    valFn = valFn,
    encodeFn = encodeFn,
    mathFn = mathFn
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // UnaryLUTOp introduces exactly 1 cycle of latency (readSync)
  // We check that when output is valid, it matches the input from the previous cycle
  when(pastValid() && dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === past(dut.io.a.stream.payload(0)), "Math LUT data mismatch")
  }
}

object MathLutsFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new MathLutsFormal, "mathluts_formal")
  }
}
