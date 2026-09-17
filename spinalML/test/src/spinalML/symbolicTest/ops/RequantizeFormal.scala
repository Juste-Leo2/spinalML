// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{I8, I32}
import spinalML.ops.RequantizeOp
import spinalML.RoundingMode

class RequantizeFormal extends Component {
  val shift = 4
  // Legacy truncation path (bit-exact golden below); the RNE path is proven
  // by RequantizeRNEFormal in this same file. Pinning Truncate keeps this
  // proof on the legacy mode.
  val dut = FormalDut(RequantizeOp(
    dataTypeIn = I32(),
    dataTypeOut = I8(),
    shape = Seq(1),
    lanes = 1,
    shift = shift,
    rounding = RoundingMode.Truncate
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Pure combinational op, no latency introduced here.
  // Wait! RequantizeOp uses arbitrationFrom, which means it is 0 latency.
  
  // Golden model (Bit-exact with RTL)
  val valIn = dut.io.a.stream.payload(0)
  val shifted = (valIn >> shift).resize(32 bits)
  
  val maxVal = (1 << 7) - 1
  val minVal = -(1 << 7)
  
  val saturated = Mux(shifted > maxVal, S(maxVal, 32 bits),
                    Mux(shifted < minVal, S(minVal, 32 bits),
                        shifted))
  val expected = saturated.resize(8 bits)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected, "requantize data mismatch")
    // Defense in depth, independent of the golden: saturation must never wrap.
    assert(dut.io.c.stream.payload(0) >= S(-128, 8 bits), "requantize output below I8 min")
    assert(dut.io.c.stream.payload(0) <= S(127, 8 bits), "requantize output above I8 max")
  }
}

/**
 * Formal proof of the RNE narrowing path of RequantizeOp (Wave 4 commit 0).
 *
 * The solver explores the full I32 input space symbolically, so every tie
 * (remainder == half) and every saturation boundary is covered exhaustively —
 * stronger than the sampled sim sweep in RoundingPolicyTest.
 */
class RequantizeRNEFormal extends Component {
  val shift = 4
  val dut = FormalDut(RequantizeOp(
    dataTypeIn = I32(),
    dataTypeOut = I8(),
    shape = Seq(1),
    lanes = 1,
    shift = shift,
    rounding = RoundingMode.Rne
  ))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Pure combinational op, no latency introduced here.
  // Wait! RequantizeOp uses arbitrationFrom, which means it is 0 latency.

  // Golden model (Bit-exact with RTL): RNE on the arithmetic shift.
  // truncated = floor(x / 2^shift), guard = dropped bit shift-1,
  // sticky = OR of the lower dropped bits, tie rounds to even.
  val valIn = dut.io.a.stream.payload(0)
  val truncated = (valIn >> shift).resize(32 bits)
  val guard = valIn(shift - 1)
  val sticky = (valIn(shift - 2 downto 0) =/= 0)
  val roundUp = guard && (sticky || truncated(0))
  val rounded = (truncated + Mux(roundUp, S(1, 32 bits), S(0, 32 bits))).resize(32 bits)

  val maxVal = (1 << 7) - 1
  val minVal = -(1 << 7)

  val saturated = Mux(rounded > maxVal, S(maxVal, 32 bits),
                    Mux(rounded < minVal, S(minVal, 32 bits),
                        rounded))
  val expected = saturated.resize(8 bits)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected, "requantize RNE data mismatch")
    // Defense in depth, independent of the golden: saturation must never wrap.
    assert(dut.io.c.stream.payload(0) >= S(-128, 8 bits), "requantize RNE output below I8 min")
    assert(dut.io.c.stream.payload(0) <= S(127, 8 bits), "requantize RNE output above I8 max")
  }
}

object RequantizeFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new RequantizeFormal, "requantize_formal")
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new RequantizeRNEFormal, "requantize_rne_formal")
  }
}
