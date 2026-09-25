// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}
import spinalML.ops.DivTestComp
import spinalML.utils.MathLUTs

/**
 * Exact integer division proof (replaces the old reciprocal-ROM + multiply
 * proof). Instead of a golden divider, this checks the defining
 * characterization of truncating division:
 *   q = trunc(a / b) <=> r = a - q*b satisfies |r| < |b| and
 *   sign(r) in {0, sign(a)}, with the engine's saturation policy on the two
 *   unreachable-by-math cases (b == 0, INT_MIN / -1).
 */
class DivFormal_I8 extends Component {
  val dut = FormalDut(DivTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  // No `b =/= 0` assumption: the div-by-zero saturation policy is proven too.

  val trackedA = Reg(Vec(I8(), 2))
  val trackedB = Reg(Vec(I8(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.b.stream.valid &&
    dut.io.a.stream.ready && dut.io.b.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedA := dut.io.a.stream.payload
    trackedB := dut.io.b.stream.payload
  }

  val w = 8
  val wide = 2 * w
  val maxVal = S((1 << (w - 1)) - 1, w bits)
  val minVal = S(-(1 << (w - 1)), w bits)

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for (i <- 0 until 2) {
      val a = trackedA(i).asInstanceOf[SInt]
      val b = trackedB(i).asInstanceOf[SInt]
      val q = dut.io.c.stream.payload(i).asInstanceOf[SInt]
      val aW = a.resize(wide bits)
      val bW = b.resize(wide bits)
      val qW = q.resize(wide bits)
      val r = aW - (qW * bW).resize(wide bits)
      val rAbs = Mux(r < 0, -r, r)
      val bAbs = Mux(bW < 0, -bW, bW)

      when(b === 0) {
        // Saturation of the mathematical infinity, sign-aware.
        assert(q === Mux(a.msb, minVal, Mux(a === 0, S(0, w bits), maxVal)),
          s"div0 saturation on lane $i")
      } otherwise {
        when(a === minVal && b === S(-1, w bits)) {
          // INT_MIN / -1: +2^(w-1) not representable -> +max.
          assert(q === maxVal, s"INT_MIN/-1 saturation on lane $i")
        } otherwise {
          assert(rAbs < bAbs, s"|remainder| must be < |divisor| on lane $i")
          assert(r === 0 || r.msb === a.msb,
            s"remainder sign must follow the dividend on lane $i")
        }
      }
    }
    hasChecked := True
  }
}

class DivFormal_FP4 extends Component {
  val dut = FormalDut(DivTestComp(FP4_E2M1()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  
  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  // Assume b is not 0 for division
  for (i <- 0 until 2) {
    assume(dut.io.b.stream.payload(i).asBits.asUInt =/= 0)
  }

  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))
  val valFn = MathLUTs.floatValFn(2, 1)
  val encodeFn = MathLUTs.floatEncodeFn(2, 1)
  val romContent = for(i <- 0 until 16) yield {
    val resDouble = mathFn(valFn(i))
    U(encodeFn(resDouble), 4 bits)
  }
  val goldenRecipRom = Mem(UInt(4 bits), initialContent = romContent)

  val expectedPayload = Vec(FP4_E2M1(), 2)
  for(i <- 0 until 2) {
    val bBits = dut.io.b.stream.payload(i).asBits.asUInt
    val goldenRecipBits = goldenRecipRom.readAsync(bBits)
    
    val goldenRecipFloat = FP4_E2M1()
    goldenRecipFloat.assignFromBits(goldenRecipBits.asBits)
    
    expectedPayload(i).assignFrom(spinalML.utils.Float.mul(dut.io.a.stream.payload(i), goldenRecipFloat).asInstanceOf[FloatML])
  }

  val trackedExpected = Reg(Vec(FP4_E2M1(), 2))
  val track = RegInit(False)
  val hasChecked = RegInit(False)

  val fireIn = dut.io.a.stream.valid && dut.io.b.stream.valid && dut.io.a.stream.ready
  when(fireIn && !track && !hasChecked) {
    track := True
    trackedExpected := expectedPayload
  }

  val fireOut = dut.io.c.stream.valid && dut.io.c.stream.ready
  when(fireOut && track && !hasChecked) {
    for(i <- 0 until 2) {
      assert(dut.io.c.stream.payload(i).asBits === trackedExpected(i).asBits, s"Div FP4 mismatch on lane $i")
    }
    hasChecked := True
  }
}

object DivFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DivFormal_I8, "div_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DivFormal_FP4, "div_fp4")
  }
}
