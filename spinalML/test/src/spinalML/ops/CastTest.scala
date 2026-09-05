// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.{BF16, FloatML, I8, I16, I32}
import spinalML.dtypes.BF16Sim
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

class CastTest extends AnyFunSuite {
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

  for ((name, dt) <- compileTypes.take(2)) {
    test(s"Test Cast dequant compilation on $name") {
      SpinalConfig().generateVerilog(CastTestComp(dt()))
      SpinalConfig().generateVerilog(CastDequantTestComp(dt(), Seq(4), 4, Seq(0.25)))
      SpinalConfig().generateVerilog(CastDequantTestComp(dt(), Seq(8), 4, Seq(0.5, 2.0)))
    }
  }
}
