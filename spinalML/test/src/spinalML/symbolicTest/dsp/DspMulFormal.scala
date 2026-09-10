// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.dsp

import spinal.core._
import spinal.core.formal._
import spinalML.dsp._

class DspMulFormal_SInt extends Component {
  val cfg = DspConfig(target = DspTarget.Generic, useDsp = true, latency = 1)
  val dut = FormalDut(DspMulSIntComp(wIn = 8, wAcc = 16, latency = 1, cfg = cfg))

  anyseq(dut.io.a)
  anyseq(dut.io.b)
  anyseq(dut.io.enable)

  assumeInitial(clockDomain.isResetActive)

  val expectedProd = (dut.io.a * dut.io.b).resized
  val trackedExpected = RegNextWhen(expectedProd, dut.io.enable, init = S(0, 16 bits))

  when(!clockDomain.isResetActive) {
    // Mathematical equivalence assertion
    assert(dut.io.result === trackedExpected, "DspMul SInt output must bit-exactly match mathematical product")
  }
}

class DspMulFormal_Combinational extends Component {
  val cfg = DspConfig(target = DspTarget.Generic, useDsp = false, latency = 0)
  val dut = FormalDut(DspMulSIntComp(wIn = 8, wAcc = 16, latency = 0, cfg = cfg))

  anyseq(dut.io.a)
  anyseq(dut.io.b)
  anyseq(dut.io.enable)

  assumeInitial(clockDomain.isResetActive)

  val expectedProd = (dut.io.a * dut.io.b).resized
  when(!clockDomain.isResetActive) {
    assert(dut.io.result === expectedProd, "DspMul combinational output must immediately match mathematical product")
  }
}

object DspMulFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DspMulFormal_SInt, "dsp_mul_sint")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DspMulFormal_Combinational, "dsp_mul_comb")
  }
}
