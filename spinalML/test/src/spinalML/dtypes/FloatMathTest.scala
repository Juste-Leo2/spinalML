package spinalML.dtypes

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinalML.utils.Float
import org.scalatest.funsuite.AnyFunSuite

// Scala helper to convert Java Float to BF16 bits and vice-versa for Testing
object BF16Sim {
  def floatToBf16Bits(f: scala.Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    // BF16 is just the top 16 bits of FP32
    (bits >>> 16) & 0xFFFF
  }
  
  def bf16BitsToFloat(bf16: Int): scala.Float = {
    val bits = (bf16 & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits)
  }
}

// Component to test purely combinatorial multiplication
case class FloatMathTestComp() extends Component {
  val a = in(BF16())
  val b = in(BF16())
  val c = out(BF16())
  
  c := Float.mul(a, b)
}

class FloatMathTest extends AnyFunSuite {
  test("Test Combinatorial BF16 Multiplication") {
    SimConfig.withWave.compile(FloatMathTestComp()).doSim { dut =>
      
      def testMul(fA: scala.Float, fB: scala.Float): Unit = {
        val bitsA = BF16Sim.floatToBf16Bits(fA)
        val bitsB = BF16Sim.floatToBf16Bits(fB)
        
        dut.a.sign #= ((bitsA >> 15) & 1) != 0
        dut.a.exponent #= (bitsA >> 7) & 0xFF
        dut.a.mantissa #= bitsA & 0x7F
        
        dut.b.sign #= ((bitsB >> 15) & 1) != 0
        dut.b.exponent #= (bitsB >> 7) & 0xFF
        dut.b.mantissa #= bitsB & 0x7F
        
        sleep(1) // combinational delay
        
        val signC = if(dut.c.sign.toBoolean) 1 else 0
        val expC = dut.c.exponent.toInt
        val mantC = dut.c.mantissa.toInt
        val bitsC = (signC << 15) | (expC << 7) | mantC
        
        val fC = BF16Sim.bf16BitsToFloat(bitsC)
        val expected = fA * fB
        
        // Allowed error is extremely small for BF16 precision bounds
        // (Just check if they are identical in BF16 cast)
        val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
        val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
        
        assert(fC == expectedBf16Float, s"Failed for $fA * $fB. Expected $expectedBf16Float, got $fC")
      }
      
      // Test cases
      testMul(1.5f, 2.0f) // 3.0
      testMul(-1.5f, 2.0f) // -3.0
      testMul(-2.5f, -2.0f) // 5.0
      testMul(0.0f, 5.0f) // 0.0
      testMul(10.25f, 0.5f) // 5.125
      testMul(12.0f, 12.0f) // 144.0
    }
  }

  test("Test Combinatorial BF16 Addition") {
    SimConfig.withWave.compile(FloatMathAddTestComp()).doSim { dut =>
      
      def testAdd(fA: scala.Float, fB: scala.Float): Unit = {
        val bitsA = BF16Sim.floatToBf16Bits(fA)
        val bitsB = BF16Sim.floatToBf16Bits(fB)
        
        dut.a.sign #= ((bitsA >> 15) & 1) != 0
        dut.a.exponent #= (bitsA >> 7) & 0xFF
        dut.a.mantissa #= bitsA & 0x7F
        
        dut.b.sign #= ((bitsB >> 15) & 1) != 0
        dut.b.exponent #= (bitsB >> 7) & 0xFF
        dut.b.mantissa #= bitsB & 0x7F
        
        sleep(1) // combinational delay
        
        val signC = if(dut.c.sign.toBoolean) 1 else 0
        val expC = dut.c.exponent.toInt
        val mantC = dut.c.mantissa.toInt
        val bitsC = (signC << 15) | (expC << 7) | mantC
        
        val fC = BF16Sim.bf16BitsToFloat(bitsC)
        val expected = fA + fB
        
        val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
        val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
        
        // Due to guard bits differences, allow a tiny epsilon difference if not exact match
        val diff = Math.abs(fC - expectedBf16Float)
        val epsilon = Math.max(Math.abs(expectedBf16Float) * 0.05, 0.01) // 5% relative or 0.01 abs error
        assert(diff <= epsilon, s"Failed for $fA + $fB. Expected $expectedBf16Float, got $fC (Diff: $diff)")
      }
      
      // Test cases
      testAdd(1.5f, 2.0f) // 3.5
      testAdd(-1.5f, 2.0f) // 0.5
      testAdd(-2.5f, -2.0f) // -4.5
      testAdd(0.0f, 5.0f) // 5.0
      testAdd(10.25f, 0.5f) // 10.75
      testAdd(1.0f, -1.0f) // 0.0
      testAdd(100.0f, 0.001f) // 100.0 (absorption)
      testAdd(0.001f, 100.0f) // 100.0
    }
  }

  test("Test Combinatorial SInt to BF16 Conversion") {
    SimConfig.withWave.compile(FloatMathFromSIntTestComp()).doSim { dut =>
      
      def testFromSInt(value: Int): Unit = {
        dut.a #= value
        
        sleep(1) // combinational delay
        
        val signC = if(dut.c.sign.toBoolean) 1 else 0
        val expC = dut.c.exponent.toInt
        val mantC = dut.c.mantissa.toInt
        val bitsC = (signC << 15) | (expC << 7) | mantC
        
        val fC = BF16Sim.bf16BitsToFloat(bitsC)
        val expected = value.toFloat
        
        val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
        val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
        
        assert(fC == expectedBf16Float, s"Failed for SInt $value. Expected $expectedBf16Float, got $fC")
      }
      
      // Test cases
      testFromSInt(5)
      testFromSInt(-5)
      testFromSInt(0)
      testFromSInt(127)
      testFromSInt(-128)
      testFromSInt(1)
      testFromSInt(-1)
      testFromSInt(42)
    }
  }

  test("Test fromDouble field conversion (golden model mirror)") {
    // (value, expBits, mantBits) -> expected (sign, biasedExp, mantissa)
    def check(value: Double, expBits: Int, mantBits: Int, eSign: Boolean, eExp: Int, eMant: Long): Unit = {
      val (sign, exp, mant) = spinalML.utils.Float.doubleToFields(value, expBits, mantBits)
      assert(sign == eSign && exp == eExp && mant == eMant,
        s"fromDouble($value, E$expBits M$mantBits): got ($sign, $exp, $mant), expected ($eSign, $eExp, $eMant)")
    }

    // Zero / NaN
    check(0.0, 8, 7, false, 0, 0)
    check(-0.0, 8, 7, false, 0, 0)
    check(Double.NaN, 8, 7, false, 0, 0)

    // BF16 (E8 M7)
    check(1.0, 8, 7, false, 127, 0)
    check(-1.0, 8, 7, true, 127, 0)
    check(2.5, 8, 7, false, 128, 32)
    check(-2.5, 8, 7, true, 128, 32)

    // Half-to-even rounding: 1 + 129/256 -> (mant-1)*128 = 64.5 -> rounds to 64 (even)
    check(1.0 + 129.0 / 256.0, 8, 7, false, 127, 64)
    // Half-to-even upward: 1 + 131/256 -> 65.5 -> rounds to 66 (even)
    check(1.0 + 131.0 / 256.0, 8, 7, false, 127, 66)

    // FP8 (E4 M3): bias = 7
    check(1.5, 4, 3, false, 7 + 0, 4)
    check(-1.5, 4, 3, true, 7 + 0, 4)
    check(10.0, 4, 3, false, 7 + 3, 2)

    // FP4 (E2 M1): bias = 1
    check(1.0, 2, 1, false, 1, 0)
    check(1.5, 2, 1, false, 1, 1)
    check(2.0, 2, 1, false, 2, 0)
    check(3.0, 2, 1, false, 2, 1)   // max normal
    check(4.0, 2, 1, false, 3, 0)   // saturate to infinity encoding
    check(-6.0, 2, 1, true, 3, 0)   // negative saturation
    check(0.75, 2, 1, false, 0, 0)  // underflow -> zero (no subnormals)
    check(-0.4, 2, 1, true, 0, 0)   // underflow -> zero
  }
}

// Component to test purely combinatorial addition
case class FloatMathAddTestComp() extends Component {
  val a = in(BF16())
  val b = in(BF16())
  val c = out(BF16())
  
  c := Float.add(a, b)
}

// Component to test combinatorial conversion from SInt to BF16
case class FloatMathFromSIntTestComp() extends Component {
  val a = in(SInt(8 bits))
  val c = out(BF16())
  
  c := Float.fromSInt(a, 8, 7) // BF16 parameters
}

// Anti-wrap regression component: exponent arithmetic must never wrap
// before the saturation checks (see Float.mul / Float.add widening fix)
case class FloatMathSaturateComp() extends Component {
  val io = new Bundle {
    val a4 = in(FP4_E2M1())
    val b4 = in(FP4_E2M1())
    val mul4 = out(FP4_E2M1())
    val a8 = in(FP8_E4M3())
    val b8 = in(FP8_E4M3())
    val add8 = out(FP8_E4M3())
    val mul8 = out(FP8_E4M3())
  }
  io.mul4 := Float.mul(io.a4, io.b4)
  io.add8 := Float.add(io.a8, io.b8)
  io.mul8 := Float.mul(io.a8, io.b8)
}

class FloatMathSaturateTest extends AnyFunSuite {
  test("Test exponent saturation (no wrap to zero/underflow)") {
    SimConfig.compile(FloatMathSaturateComp()).doSim { dut =>
      def set4(p: FloatML, sign: Boolean, exp: Int, mant: Int): Unit = { p.sign #= sign; p.exponent #= exp; p.mantissa #= mant }
      def getBits(p: FloatML): Int =
        ((if (p.sign.toBoolean) 1 else 0) << (p.exponent.getWidth + p.mantissa.getWidth)) |
         ((p.exponent.toInt << p.mantissa.getWidth)) | p.mantissa.toInt

      // FP4: 3.0 * 3.0 = 9.0 -> saturate +inf (expSum 2+2-1+1 = 4 wrapped to -4 before the fix)
      set4(dut.io.a4, false, 2, 1)
      set4(dut.io.b4, false, 2, 1)
      sleep(1)
      assert(getBits(dut.io.mul4) == 0x6, s"FP4 3*3 should saturate to +inf, got ${getBits(dut.io.mul4).toBinaryString}")

      // FP4: 2.0 * -inf -> -inf
      set4(dut.io.a4, false, 2, 0)
      set4(dut.io.b4, true, 3, 0)
      sleep(1)
      assert(getBits(dut.io.mul4) == 0xE, s"FP4 2*-inf should be -inf, got ${getBits(dut.io.mul4).toBinaryString}")

      // FP8: inf + inf -> +inf (newExp 15+1 wrapped negative before the fix)
      dut.io.a8.sign #= false; dut.io.a8.exponent #= 15; dut.io.a8.mantissa #= 0
      dut.io.b8.sign #= false; dut.io.b8.exponent #= 15; dut.io.b8.mantissa #= 0
      sleep(1)
      assert(getBits(dut.io.add8) == 0x78, s"FP8 inf+inf should be +inf, got ${getBits(dut.io.add8).toBinaryString}")

      // FP8: max normal * max normal -> saturate +inf
      dut.io.a8.sign #= false; dut.io.a8.exponent #= 14; dut.io.a8.mantissa #= 7
      dut.io.b8.sign #= false; dut.io.b8.exponent #= 14; dut.io.b8.mantissa #= 7
      sleep(1)
      assert(getBits(dut.io.mul8) == 0x78, s"FP8 240*240 should saturate to +inf, got ${getBits(dut.io.mul8).toBinaryString}")
    }
  }
}
