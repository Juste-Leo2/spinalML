// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dtypes

import spinal.core._
import spinal.core.sim._
import spinalML.utils.Float
import org.scalatest.funsuite.AnyFunSuite

// Wave 5 NaN propagation probe: E4M3 single slot (15, 7) and FP4 (3, 1)
// flow through mul/add/gt/max/roundTo/widen with the canonical encoding
// and the first-NaN sign rule. NaN is never emitted spontaneously
// (saturation still yields 448/inf); LUT/PWL ops keep mapping NaN to 0
// (documented trap, see docs/rounding_policy.md §5).
case class FloatNaNComp() extends Component {
  val io = new Bundle {
    val a4 = in(FP4_E2M1())
    val b4 = in(FP4_E2M1())
    val mul4 = out(FP4_E2M1())
    val a8 = in(FP8_E4M3())
    val b8 = in(FP8_E4M3())
    val mul8 = out(FP8_E4M3())
    val add8 = out(FP8_E4M3())
    val gt8 = out(Bool())
    val max8 = out(FP8_E4M3())
    val r8 = in(FP8_E4M3())
    val rt8 = out(FP8_E4M3())
    val w8 = out(BF16())
    val bf = in(BF16())
    val rb = out(FP8_E4M3())
  }
  io.mul4 := Float.mul(io.a4, io.b4)
  io.mul8 := Float.mul(io.a8, io.b8)
  io.add8 := Float.add(io.a8, io.b8)
  io.gt8 := Float.gt(io.a8, io.b8)
  io.max8 := Float.max(io.a8, io.b8)
  io.rt8 := Float.roundTo(io.r8, 4, 3)
  io.w8 := Float.widen(io.r8, 8, 7)
  io.rb := Float.roundTo(io.bf, 4, 3)
}

class FloatNaNTest extends AnyFunSuite {
  test("NaN operands propagate canonically, comparisons stay False") {
    SimConfig.compile(FloatNaNComp()).doSim { dut =>
      def set(p: FloatML, sign: Boolean, exp: Int, mant: Int): Unit = {
        p.sign #= sign; p.exponent #= exp; p.mantissa #= mant
      }
      def bits(p: FloatML): Int =
        ((if (p.sign.toBoolean) 1 else 0) << (p.exponent.getWidth + p.mantissa.getWidth)) |
          (p.exponent.toInt << p.mantissa.getWidth) | p.mantissa.toInt

      // mul: NaN x finite -> NaN, first-NaN sign rule
      set(dut.io.a8, false, 15, 7); set(dut.io.b8, false, 8, 0) // NaN x 2.0
      sleep(1)
      assert(bits(dut.io.mul8) == 0x7F, f"mul(NaN,2) = ${bits(dut.io.mul8)}%02x, want 7f")
      set(dut.io.a8, false, 8, 0); set(dut.io.b8, true, 15, 7) // 2.0 x -NaN
      sleep(1)
      assert(bits(dut.io.mul8) == 0xFF, f"mul(2,-NaN) = ${bits(dut.io.mul8)}%02x, want ff")

      // mul: NaN beats zero (zero class never swallows NaN)
      set(dut.io.a8, true, 15, 7); set(dut.io.b8, false, 0, 0)
      sleep(1)
      assert(bits(dut.io.mul8) == 0xFF, f"mul(-NaN,0) = ${bits(dut.io.mul8)}%02x, want ff")

      // mul FP4: NaN slot (3, 1) propagates
      set(dut.io.a4, false, 3, 1); set(dut.io.b4, false, 2, 1) // NaN x 3.0
      sleep(1)
      assert(bits(dut.io.mul4) == 0x7, f"FP4 mul(NaN,3) = ${bits(dut.io.mul4)}%x, want 7")

      // add: NaN beats the infinity path
      set(dut.io.a8, false, 15, 7); set(dut.io.b8, false, 15, 0) // NaN + inf
      sleep(1)
      assert(bits(dut.io.add8) == 0x7F, f"add(NaN,inf) = ${bits(dut.io.add8)}%02x, want 7f")

      // gt: every comparison with NaN is False
      set(dut.io.a8, false, 15, 7); set(dut.io.b8, false, 7, 0) // NaN vs 1.0
      sleep(1)
      assert(!dut.io.gt8.toBoolean, "gt(NaN,1) must be False")
      set(dut.io.a8, false, 7, 0); set(dut.io.b8, false, 15, 7) // 1.0 vs NaN
      sleep(1)
      assert(!dut.io.gt8.toBoolean, "gt(1,NaN) must be False")

      // max inherits Mux(gt, a, b): second operand wins on NaN (documented asymmetry)
      set(dut.io.a8, false, 15, 7); set(dut.io.b8, false, 7, 0)
      sleep(1)
      assert(bits(dut.io.max8) == 0x38, f"max(NaN,1) = ${bits(dut.io.max8)}%02x, want 38")
      set(dut.io.a8, false, 7, 0); set(dut.io.b8, false, 15, 7)
      sleep(1)
      assert(bits(dut.io.max8) == 0x7F, f"max(1,NaN) = ${bits(dut.io.max8)}%02x, want 7f")

      // roundTo E4M3 -> E4M3 preserves the slot
      set(dut.io.r8, true, 15, 7)
      sleep(1)
      assert(bits(dut.io.rt8) == 0xFF, f"roundTo(-NaN) = ${bits(dut.io.rt8)}%02x, want ff")

      // widen E4M3 -> BF16: canonical BF16 NaN (255, 1), sign preserved
      sleep(1)
      assert(dut.io.w8.sign.toBoolean, "widen(-NaN) keeps sign")
      assert(dut.io.w8.exponent.toInt == 255, "widen(-NaN) exponent")
      assert(dut.io.w8.mantissa.toInt == 1, "widen(-NaN) canonical mantissa 1")

      // roundTo BF16 (255, 5) -> E4M3: foreign NaN folds to the E4M3 slot
      dut.io.bf.sign #= false; dut.io.bf.exponent #= 255; dut.io.bf.mantissa #= 5
      sleep(1)
      assert(bits(dut.io.rb) == 0x7F, f"roundTo(BF16-NaN) = ${bits(dut.io.rb)}%02x, want 7f")

      // sanity: finite paths intact (2.0 x 3.0 = 6.0, 1.0 + 1.0 = 2.0)
      set(dut.io.a8, false, 8, 0); set(dut.io.b8, false, 7, 4) // 2.0 x 1.5
      sleep(1)
      assert(bits(dut.io.mul8) == 0x44, f"mul(2,1.5) = ${bits(dut.io.mul8)}%02x, want 44")
      set(dut.io.a8, false, 7, 0); set(dut.io.b8, false, 7, 0) // 1.0 + 1.0
      sleep(1)
      assert(bits(dut.io.add8) == 0x40, f"add(1,1) = ${bits(dut.io.add8)}%02x, want 40")
    }
  }
}
