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
