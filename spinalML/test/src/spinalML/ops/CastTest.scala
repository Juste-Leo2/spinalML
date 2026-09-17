// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.{BF16, FloatML, FP8_E4M3, I8, I16, I32}
import spinalML.dtypes.BF16Sim
import spinalML.{RoundingMode, RoundingConfig}
import org.scalatest.funsuite.AnyFunSuite

case class CastTestComp[TIn <: Data](dataTypeIn: HardType[TIn]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, Seq(4), lanes = 4))
    val c = master(Tensor(BF16(), Seq(4), lanes = 4))
  }
  
  val casted = cast(io.a, BF16())
  io.c <> casted
}

// Per-tensor or per-channel dequantizing cast (scales indexed by stream beat order)
case class CastDequantTestComp[TIn <: Data](dataTypeIn: HardType[TIn], shape: Seq[Int], lanes: Int, scales: Seq[Double]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, shape, lanes))
    val c = master(Tensor(BF16(), shape, lanes))
  }
  
  val casted = cast(io.a, BF16(), scales)
  io.c <> casted
}

// DTYPE-06: narrow-float targets where the mantissa window drops bits.
case class CastFP8TestComp[TIn <: Data](dataTypeIn: HardType[TIn], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, Seq(4), lanes = 4))
    val c = master(Tensor(FP8_E4M3(), Seq(4), lanes = 4))
  }

  io.c <> cast(io.a, FP8_E4M3(), rounding = rounding)
}

case class CastI16BF16TestComp(rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I16(), Seq(4), lanes = 4))
    val c = master(Tensor(BF16(), Seq(4), lanes = 4))
  }

  io.c <> cast(io.a, BF16(), rounding = rounding)
}

// OPS-10: float-source casts (elaboration throws before the fix).
case class CastF2FTestComp(rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(BF16(), Seq(4), lanes = 4))
    val c = master(Tensor(FP8_E4M3(), Seq(4), lanes = 4))
  }

  io.c <> cast(io.a, FP8_E4M3(), rounding = rounding)
}

case class CastF2ITestComp(rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(BF16(), Seq(4), lanes = 4))
    val c = master(Tensor(I8(), Seq(4), lanes = 4))
  }

  io.c <> cast(io.a, I8(), rounding = rounding)
}

class CastTest extends AnyFunSuite {
  // OPS-10: BF16(140.5) -> FP8. Fraction 0001101b over 7 bits narrows to
  // 3 bits: kept 000, dropped 1101 (guard=1, sticky=1) -> RNE 136.
  // (Stays under the E4M3 448 ceiling so no saturation interferes.)
  test("OPS-10 Cast BF16 to FP8 narrows with RNE to 136") {
    SimConfig.withWave.compile(CastF2FTestComp(RoundingMode.Rne)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0).asInstanceOf[FloatML].sign #= false
      dut.io.a.stream.payload(0).asInstanceOf[FloatML].exponent #= 134
      dut.io.a.stream.payload(0).asInstanceOf[FloatML].mantissa #= 13
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for float->float cast")
      val o = dut.io.c.stream.payload(0).asInstanceOf[FloatML]
      assert(!o.sign.toBoolean && o.exponent.toInt == 14 && o.mantissa.toInt == 1,
        s"expected 136 (exp=14 mant=1), got sign=${o.sign.toBoolean} exp=${o.exponent.toInt} mant=${o.mantissa.toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  // OPS-10: same input under Truncate keeps the window (128).
  test("OPS-10 Cast BF16 to FP8 truncates to 128 (Truncate legacy)") {
    SimConfig.withWave.compile(CastF2FTestComp(RoundingMode.Truncate)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0).asInstanceOf[FloatML].sign #= false
      dut.io.a.stream.payload(0).asInstanceOf[FloatML].exponent #= 134
      dut.io.a.stream.payload(0).asInstanceOf[FloatML].mantissa #= 13
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for float->float cast")
      val o = dut.io.c.stream.payload(0).asInstanceOf[FloatML]
      assert(!o.sign.toBoolean && o.exponent.toInt == 14 && o.mantissa.toInt == 0,
        s"expected 128 (exp=14 mant=0), got sign=${o.sign.toBoolean} exp=${o.exponent.toInt} mant=${o.mantissa.toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  // OPS-10: BF16 -> I8 with RNE ties (43.5 -> 44), saturation (1032 -> 127),
  // negatives (-43.5 -> -44) and the exact minimum (-128.0 -> -128).
  test("OPS-10 Cast BF16 to I8 rounds ties and saturates (RNE)") {
    SimConfig.withWave.compile(CastF2ITestComp(RoundingMode.Rne)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // (sign, exp, mant, expected I8)
      val vectors = Seq(
        (false, 132, 46, 44),    // 43.5 -> 44 (tie, even up)
        (false, 137, 16, 127),   // 1152.0 -> saturate 127
        (true, 132, 46, -44),    // -43.5 -> -44
        (true, 134, 0, -128)     // -128.0 exact minimum
      )
      for ((s, e, m, expected) <- vectors) {
        val in = dut.io.a.stream.payload(0).asInstanceOf[FloatML]
        in.sign #= s
        in.exponent #= e
        in.mantissa #= m
        dut.io.a.stream.valid #= true
        assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
          s"no output for float->int cast ($s,$e,$m)")
        val got = dut.io.c.stream.payload(0).asInstanceOf[SInt].toInt
        dut.io.a.stream.valid #= false
        dut.clockDomain.waitSampling(2)
        assert(got == expected, s"($s,$e,$m): expected $expected, got $got")
      }
      dut.clockDomain.waitSampling(5)
    }
  }

  test("DTYPE-06 Cast I8 127 to FP8 rounds up with carry to 128 (RNE)") {
    SimConfig.withWave.compile(CastFP8TestComp(I8(), RoundingMode.Rne)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0).asInstanceOf[SInt] #= 127
      dut.io.a.stream.valid #= true
      // Combinational op: bounded wait, never an infinite load.
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for cast input")
      val o = dut.io.c.stream.payload(0).asInstanceOf[FloatML]
      assert(!o.sign.toBoolean && o.exponent.toInt == 14 && o.mantissa.toInt == 0,
        s"expected +128 (exp=14 mant=0), got sign=${o.sign.toBoolean} exp=${o.exponent.toInt} mant=${o.mantissa.toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  // DTYPE-06: same input under Truncate keeps the legacy window (120).
  test("DTYPE-06 Cast I8 127 to FP8 truncates to 120 (Truncate legacy)") {
    SimConfig.withWave.compile(CastFP8TestComp(I8(), RoundingMode.Truncate)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0).asInstanceOf[SInt] #= 127
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for cast input")
      val o = dut.io.c.stream.payload(0).asInstanceOf[FloatML]
      assert(!o.sign.toBoolean && o.exponent.toInt == 13 && o.mantissa.toInt == 7,
        s"expected 120 (exp=13 mant=7), got sign=${o.sign.toBoolean} exp=${o.exponent.toInt} mant=${o.mantissa.toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  // DTYPE-06: tie-to-even (I16 1036 -> BF16 1040, kept LSB odd rounds up)
  // and carry into the exponent (I16 32767 -> BF16 32768).
  test("DTYPE-06 Cast I16 to BF16 rounds ties to even with carry (RNE)") {
    SimConfig.withWave.compile(CastI16BF16TestComp(RoundingMode.Rne)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      def convert(v: Int): (Boolean, Int, Int) = {
        dut.io.a.stream.payload(0).asInstanceOf[SInt] #= v
        dut.io.a.stream.valid #= true
        assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
          s"no output for cast input $v")
        val o = dut.io.c.stream.payload(0).asInstanceOf[FloatML]
        dut.io.a.stream.valid #= false
        dut.clockDomain.waitSampling(2)
        (o.sign.toBoolean, o.exponent.toInt, o.mantissa.toInt)
      }

      val tie = convert(1036)
      assert(tie == (false, 137, 2), s"tie 1036 -> 1040 (exp=137 mant=2), got $tie")
      // Guard 0 with odd kept LSB stays put (off-by-one guard selection
      // would wrongly round this up to 1040).
      val down = convert(1032)
      assert(down == (false, 137, 1), s"1032 exact (exp=137 mant=1), got $down")
      val ovf = convert(32767)
      assert(ovf == (false, 142, 0), s"32767 -> 32768 (exp=142 mant=0), got $ovf")
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test streaming Cast operation SInt -> BF16") {
    SimConfig.withWave.compile(CastTestComp(I8())).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      val inputData = Array(
        Array(5, -5, 0, 127),
        Array(-128, 1, -1, 42)
      )
      
      var outputIndex = 0
      val numExpectedOutputs = inputData.length
      
      // Monitor output
      fork {
        while (outputIndex < numExpectedOutputs) {
          dut.clockDomain.waitSampling()
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            
            val expectedInputs = inputData(outputIndex)
            for (i <- 0 until 4) {
              val signC = if(dut.io.c.stream.payload(i).asInstanceOf[FloatML].sign.toBoolean) 1 else 0
              val expC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].exponent.toInt
              val mantC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].mantissa.toInt
              val bitsC = (signC << 15) | (expC << 7) | mantC
              
              val fC = BF16Sim.bf16BitsToFloat(bitsC)
              val expected = expectedInputs(i).toFloat
              
              val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
              val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
              
              assert(fC == expectedBf16Float, s"Index $outputIndex Lane $i: Expected $expectedBf16Float, got $fC")
            }
            outputIndex += 1
          }
        }
      }
      
      // Drive inputs
      for (i <- 0 until inputData.length) {
        dut.io.a.stream.valid #= true
        for (lane <- 0 until 4) {
          dut.io.a.stream.payload(lane).asInstanceOf[SInt] #= inputData(i)(lane)
        }
        
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
        // Random wait between beats
        dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
      }
      
      // Wait for all outputs to be checked
      dut.clockDomain.waitSamplingWhere(outputIndex == numExpectedOutputs)
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test streaming Dequant Cast SInt -> BF16 with per-tensor scale") {
    SimConfig.withWave.compile(CastDequantTestComp(I8(), Seq(4), 4, Seq(0.5))).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      val inputData = Array(
        Array(2, -2, 4, 8),
        Array(127, -128, 10, -10)
      )
      val scale = 0.5f
      
      var outputIndex = 0
      val numExpectedOutputs = inputData.length
      
      fork {
        while (outputIndex < numExpectedOutputs) {
          dut.clockDomain.waitSampling()
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            
            val expectedInputs = inputData(outputIndex)
            for (i <- 0 until 4) {
              val signC = if(dut.io.c.stream.payload(i).asInstanceOf[FloatML].sign.toBoolean) 1 else 0
              val expC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].exponent.toInt
              val mantC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].mantissa.toInt
              val bitsC = (signC << 15) | (expC << 7) | mantC
              
              val fC = BF16Sim.bf16BitsToFloat(bitsC)
              val expected = expectedInputs(i).toFloat * scale
              
              val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
              val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
              
              assert(fC == expectedBf16Float, s"Index $outputIndex Lane $i: Expected $expectedBf16Float, got $fC")
            }
            outputIndex += 1
          }
        }
      }
      
      for (i <- 0 until inputData.length) {
        dut.io.a.stream.valid #= true
        for (lane <- 0 until 4) {
          dut.io.a.stream.payload(lane).asInstanceOf[SInt] #= inputData(i)(lane)
        }
        
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
        dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
      }
      
      dut.clockDomain.waitSamplingWhere(outputIndex == numExpectedOutputs)
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test streaming Dequant Cast SInt -> BF16 with per-channel scales") {
    // shape (8), lanes 4 -> 2 stream beats = 2 channels
    SimConfig.withWave.compile(CastDequantTestComp(I8(), Seq(8), 4, Seq(0.5, 2.0))).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      // Beat 0 scaled by 0.5, beat 1 scaled by 2.0
      val inputData = Array(
        Array(2, -2, 4, -4),
        Array(1, 2, 3, -1),
        Array(6, -6, 8, -8),
        Array(5, 10, 15, -20)
      )
      val beatScales = Seq(0.5f, 2.0f)
      
      var outputIndex = 0
      val numExpectedOutputs = inputData.length
      
      fork {
        while (outputIndex < numExpectedOutputs) {
          dut.clockDomain.waitSampling()
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            
            val expectedInputs = inputData(outputIndex)
            val beatScale = beatScales(outputIndex % beatScales.length)
            for (i <- 0 until 4) {
              val signC = if(dut.io.c.stream.payload(i).asInstanceOf[FloatML].sign.toBoolean) 1 else 0
              val expC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].exponent.toInt
              val mantC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].mantissa.toInt
              val bitsC = (signC << 15) | (expC << 7) | mantC
              
              val fC = BF16Sim.bf16BitsToFloat(bitsC)
              val expected = expectedInputs(i).toFloat * beatScale
              
              val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
              val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
              
              assert(fC == expectedBf16Float, s"Beat $outputIndex Lane $i: Expected $expectedBf16Float, got $fC")
            }
            outputIndex += 1
          }
        }
      }
      
      for (i <- 0 until inputData.length) {
        dut.io.a.stream.valid #= true
        for (lane <- 0 until 4) {
          dut.io.a.stream.payload(lane).asInstanceOf[SInt] #= inputData(i)(lane)
        }
        
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
        dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
      }
      
      dut.clockDomain.waitSamplingWhere(outputIndex == numExpectedOutputs)
      dut.clockDomain.waitSampling(5)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("I16", () => I16()),
    ("I32", () => I32())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test Cast compilation on $name") {
      SpinalConfig().generateVerilog(CastTestComp(dt()))
    }
  }

  test("Test CastF2F compilation (BF16 -> FP8)") {
    SpinalConfig().generateVerilog(CastF2FTestComp())
  }

  test("Test CastF2I compilation (BF16 -> I8)") {
    SpinalConfig().generateVerilog(CastF2ITestComp())
  }

  for ((name, dt) <- compileTypes.take(2)) {
    test(s"Test Cast dequant compilation on $name") {
      SpinalConfig().generateVerilog(CastTestComp(dt()))
      SpinalConfig().generateVerilog(CastDequantTestComp(dt(), Seq(4), 4, Seq(0.25)))
      SpinalConfig().generateVerilog(CastDequantTestComp(dt(), Seq(8), 4, Seq(0.5, 2.0)))
    }
  }

  test("Test Cast with runtime dynamic scale port") {
    case class CastRuntimeScaleTestComp[TIn <: Data](dataTypeIn: HardType[TIn], shape: Seq[Int], lanes: Int) extends Component {
      val io = new Bundle {
        val a = slave(Tensor(dataTypeIn, shape, lanes))
        val c = master(Tensor(BF16(), shape, lanes))
        val scale = in Bits(32 bits)
      }
      val casted = cast(io.a, BF16(), runtimeScalePort = Some(io.scale))
      io.c <> casted
    }

    SpinalConfig().generateVerilog(CastRuntimeScaleTestComp(I8(), Seq(4), 4))
  }
}
