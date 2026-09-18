// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.poolings

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, U8, FP8_E4M3, I16, BF16}
import spinalML.{RoundingConfig, RoundingMode}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the AvgPool1D operation
case class AvgPool1DTestComp[T <: Data](dataType: HardType[T], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 2))
  }
  
  // poolSize = 2, stride = 2
  io.c <> spinalML.poolings.avgpool1d(io.a, poolSize = 2, stride = 2, rounding = rounding)
}

// ACT-01 repro: L=7, pool=2, stride=2 -> L_out=3 leaves one tail element per
// frame that must be drained before the next frame starts.
case class AvgPool1DResidueTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(7, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(3, 2), lanes = 2))
  }

  io.c <> spinalML.poolings.avgpool1d(io.a, poolSize = 2, stride = 2)
}

class AvgPool1DTest extends AnyFunSuite {
  test("Test streaming AvgPool1D operation on I8 tensors") {
    SimConfig.withWave.compile(AvgPool1DTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send sequence: (5, 8), (9, 4), (2, 2), (4, 6)
      val seq0 = Array(5, 9, 2, 4)
      val seq1 = Array(8, 4, 2, 6)
      var i = 0
      
      // Feed data
      fork {
        while (i < 4) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= seq0(i)
          dut.io.a.stream.payload(1) #= seq1(i)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          i += 1
        }
        dut.io.a.stream.valid #= false
      }
      
      // Check results
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      // avg(5, 9) = 7, avg(8, 4) = 6
      assert(dut.io.c.stream.payload(0).toInt == 7)
      assert(dut.io.c.stream.payload(1).toInt == 6)
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      // avg(2, 4) = 3, avg(2, 6) = 4
      assert(dut.io.c.stream.payload(0).toInt == 3)
      assert(dut.io.c.stream.payload(1).toInt == 4)
      
      dut.clockDomain.waitSampling(5)
    }
  }

  // Deterministic tie frames for the int shift. SInt: window 0 = (7, 8) = 7.5
  // (floor 7 odd -> RNE 8) and window 1 = (-2, -3) = -2.5 (floor -3 odd -> RNE -2).
  // UInt: window 0 = (7, 8) = 7.5 (floor 7 odd -> RNE 8) and window 1 =
  // (2, 3) = 2.5 (floor 2 even -> stays 2). The legacy truncate lane must keep
  // the arithmetic-shift floor.
  private def runAvgPool1DTie[T <: Data](dataType: HardType[T], rounding: RoundingMode,
                                         seq0: Seq[Int], expected: (Int, Int)): Unit = {
    SimConfig.compile(AvgPool1DTestComp(dataType, rounding)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      val seq1 = Array(0, 0, 0, 0)
      var i = 0
      fork {
        while (i < 4) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0).asInstanceOf[BaseType].assignBigInt(BigInt(seq0(i)))
          dut.io.a.stream.payload(1).asInstanceOf[BaseType].assignBigInt(BigInt(seq1(i)))
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          i += 1
        }
        dut.io.a.stream.valid #= false
      }

      for (o <- 0 until 2) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
        val result = dut.io.c.stream.payload(0).asInstanceOf[BaseType].toBigInt.toInt
        val exp = if (o == 0) expected._1 else expected._2
        assert(result == exp, s"Output $o: expected $exp, got $result (rounding=$rounding)")
      }
      dut.clockDomain.waitSampling(5)
    }
  }

  test("AvgPool1D SInt shift rounds ties to even in RNE mode") {
    runAvgPool1DTie(I8(), RoundingMode.Rne, Seq(7, 8, -2, -3), (8, -2))
  }

  test("AvgPool1D SInt shift truncates in Truncate mode (legacy)") {
    runAvgPool1DTie(I8(), RoundingMode.Truncate, Seq(7, 8, -2, -3), (7, -3))
  }

  test("AvgPool1D UInt shift rounds ties to even in RNE mode") {
    runAvgPool1DTie(U8(), RoundingMode.Rne, Seq(7, 8, 2, 3), (8, 2))
  }

  test("AvgPool1D UInt shift truncates in Truncate mode (legacy)") {
    runAvgPool1DTie(U8(), RoundingMode.Truncate, Seq(7, 8, 2, 3), (7, 2))
  }

  test("Test AvgPool1D compilation on I8") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(I8()))
  }

  test("Test AvgPool1D compilation on FP8") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(FP8_E4M3()))
  }

  test("Test AvgPool1D compilation on I16") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(I16()))
  }

  test("Test AvgPool1D compilation on BF16") {
    SpinalConfig().generateVerilog(AvgPool1DTestComp(BF16()))
  }

  test("Test AvgPool1D residue compilation on I8") {
    SpinalConfig().generateVerilog(AvgPool1DResidueTestComp(I8()))
  }
}
