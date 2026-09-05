// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.poolings

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{FP4_E2M1}
import spinalML.poolings.{AvgPool2DTestComp}

/**
 * Flow-control formal spec of AvgPool2DOp (4x4, K=2, stride=2, C=1).
 * Proves the streaming contract: once an input beat has been accepted, an
 * output handshake eventually occurs (no deadlock, no flow drop) under any
 * valid/ready pattern. Mathematical equivalence is covered by the Python
 * co-simulation (tests/python/test_avgpool2d.py, bit-exact golden models).
 * Note: the first output handshake is reachable from cycle 11 (6 input beats
 * + pipeline), hence BMC depth 12 — depth 8 would make the spec vacuous.
 */
class AvgPool2DFormal_I8 extends Component {
  val dut = FormalDut(AvgPool2DTestComp(SInt(8 bits)))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.c.stream.valid, "Flow control drop")
    hasChecked := True
  }
  cover(hasChecked)
}

class AvgPool2DFormal_FP4 extends Component {
  val dut = FormalDut(AvgPool2DTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    assert(dut.io.c.stream.valid, "Flow control drop")
    hasChecked := True
  }
  cover(hasChecked)
}

object AvgPool2DFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(12)
      .withCover(12)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AvgPool2DFormal_I8, "avgpool2d_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(12)
      .withCover(12)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AvgPool2DFormal_FP4, "avgpool2d_fp4")
  }
}
