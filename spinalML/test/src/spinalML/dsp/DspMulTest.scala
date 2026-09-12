// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dsp

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.{I8, I16, U8}

// Test harness for SInt multiplier
case class DspMulSIntComp(wIn: Int, wAcc: Int, latency: Int, cfg: DspConfig) extends Component {
  val io = new Bundle {
    val a = in SInt(wIn bits)
    val b = in SInt(wIn bits)
    val enable = in Bool()
    val result = out SInt(wAcc bits)
  }
  io.result := DspMul(io.a, io.b, enable = io.enable, accType = HardType(SInt(wAcc bits)), latency = latency, dspConfig = cfg)
}

// Test harness for UInt multiplier
case class DspMulUIntComp(wIn: Int, wAcc: Int, latency: Int, cfg: DspConfig) extends Component {
  val io = new Bundle {
    val a = in UInt(wIn bits)
    val b = in UInt(wIn bits)
    val enable = in Bool()
    val result = out UInt(wAcc bits)
  }
  io.result := DspMul(io.a, io.b, enable = io.enable, accType = HardType(UInt(wAcc bits)), latency = latency, dspConfig = cfg)
}

class DspMulTest extends AnyFunSuite {

  test("DspMul: SInt signedness and latency=1 verification") {
    val cfg = DspConfig(target = DspTarget.Generic, useDsp = true, latency = 1)
    SimConfig.compile(DspMulSIntComp(8, 16, latency = 1, cfg)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a #= 0
      dut.io.b #= 0
      dut.io.enable #= false
      dut.clockDomain.waitSampling(5) // Wait for reset release

      def testProduct(a: Int, b: Int, expected: Int): Unit = {
        dut.io.a #= a
        dut.io.b #= b
        dut.io.enable #= true
        dut.clockDomain.waitSampling(2) // 1 cycle input-to-reg + 1 cycle output settle
        assert(dut.io.result.toInt == expected, s"For ($a * $b): expected $expected, got ${dut.io.result.toInt}")
      }

      // Vector 1: Positive x Positive
      testProduct(12, 10, 120)

      // Vector 2: Positive x Negative
      testProduct(7, -5, -35)

      // Vector 3: Negative x Negative
      testProduct(-6, -8, 48)

      // Vector 4: Gowin critical corner cases (Sign saturation edge tests)
      testProduct(-128, 1, -128)
      testProduct(-128, -1, 128)
      testProduct(127, 127, 16129)

      // Enable gating: when enable = false, previous product must be held
      dut.io.enable #= false
      dut.io.a #= 10
      dut.io.b #= 10
      dut.clockDomain.waitSampling(2)
      assert(dut.io.result.toInt == 16129, s"Gating failed: expected held 16129, got ${dut.io.result.toInt}")
    }
  }

  test("DspMul: UInt unsigned arithmetic") {
    val cfg = DspConfig(target = DspTarget.Generic, useDsp = true, latency = 1)
    SimConfig.compile(DspMulUIntComp(8, 16, latency = 1, cfg)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a #= 0
      dut.io.b #= 0
      dut.io.enable #= true
      dut.clockDomain.waitSampling(5)

      def testProduct(a: Int, b: Int, expected: Int): Unit = {
        dut.io.a #= a
        dut.io.b #= b
        dut.clockDomain.waitSampling(2)
        assert(dut.io.result.toInt == expected, s"Expected $expected, got ${dut.io.result.toInt}")
      }

      testProduct(200, 10, 2000)
      testProduct(255, 255, 65025)
      testProduct(0, 255, 0)
    }
  }

  test("DspMul: Combinational latency=0 mode") {
    val cfg = DspConfig(target = DspTarget.Generic, useDsp = false, latency = 0)
    SimConfig.compile(DspMulSIntComp(8, 16, latency = 0, cfg)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.enable #= true
      dut.io.a #= 15
      dut.io.b #= -3
      sleep(1) // Combinational: valid within same cycle
      assert(dut.io.result.toInt == -45, s"Combinational mismatch: got ${dut.io.result.toInt}")
    }
  }

  test("DspMul: Bit-exact equivalence between Generic, Gowin target and No-DSP fallback") {
    val testVectors = Seq(
      (12, 10),
      (-128, 1),
      (-128, -1),
      (127, -2),
      (-50, -50),
      (0, 100),
      (100, 0)
    )

    val configs = Seq(
      DspConfig(target = DspTarget.Generic, useDsp = true, latency = 1),
      DspConfig(target = DspTarget.Gowin, useDsp = true, latency = 1),
      DspConfig(target = DspTarget.Generic, useDsp = false, latency = 1), // --no-dsp mode
      DspConfig.asic(PdkFamily.Sky130, latency = 1),
      DspConfig(target = Target.Simulation, latency = 1)
    )

    for (cfg <- configs) {
      SimConfig.compile(DspMulSIntComp(8, 16, latency = 1, cfg)).doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)
        dut.io.enable #= true
        dut.clockDomain.waitSampling(5)

        for ((a, b) <- testVectors) {
          dut.io.a #= a
          dut.io.b #= b
          dut.clockDomain.waitSampling(2)
          val expected = a * b
          assert(dut.io.result.toInt == expected, s"Config $cfg failed for $a * $b: expected $expected, got ${dut.io.result.toInt}")
        }
      }
    }
  }

  test("HardwareMul: Target.ASIC generates pure behavioral Verilog without vendor blackboxes") {
    val targetDir = "out/test_asic_mul"
    SpinalConfig(targetDirectory = targetDir).generateVerilog(
      DspMulSIntComp(8, 16, latency = 1, DspConfig.asic(PdkFamily.Sky130, latency = 1))
    )
    val content = scala.io.Source.fromFile(s"$targetDir/DspMulSIntComp.v").mkString
    assert(!content.contains("MULT18X18"), "ASIC generated Verilog must NEVER contain Gowin MULT18X18")
  }

  test("HardwareMul: Target.FPGA(Gowin) synthesizes MULT18X18 hardware macro") {
    val targetDir = "out/test_gowin_mul"
    SpinalConfig(targetDirectory = targetDir).generateVerilog(
      DspMulSIntComp(8, 16, latency = 1, DspConfig.fpga(FpgaFamily.Gowin, useDsp = true, latency = 1))
    )
    val content = scala.io.Source.fromFile(s"$targetDir/DspMulSIntComp.v").mkString
    assert(content.contains("MULT18X18"), "Gowin target must instantiate MULT18X18 hard macro")
  }
}

