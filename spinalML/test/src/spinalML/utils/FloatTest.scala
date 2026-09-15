// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.{FloatML, FP4_E2M1}

// FP8_E4M3 (4, 3) -> FP8_E5M2 (5, 2): exactly one mantissa bit is dropped
case class FloatRoundToDrop1TestComp() extends Component {
  val io = new Bundle {
    val a = in(FloatML(4, 3))
    val c = out(FloatML(5, 2))
  }
  io.c := Float.roundTo(io.a, 5, 2)
}

// FP4_E2M1 has a single mantissa bit: mantBits - 2 == -1 in the sticky term
case class FloatMulFp4TestComp() extends Component {
  val io = new Bundle {
    val a = in(FP4_E2M1())
    val b = in(FP4_E2M1())
    val c = out(FP4_E2M1())
  }
  io.c := Float.mul(io.a, io.b)
}

// FP4 (2 bits of exponent, bias 1) -> FP32 (8 bits, bias 127): biasDelta = 126,
// the widened exponent sum must not be truncated to the input width.
case class FloatRoundToWidenExpTestComp() extends Component {
  val io = new Bundle {
    val a = in(FP4_E2M1())
    val c = out(FloatML(8, 23))
  }
  val aReg = RegNext(io.a)
  io.c := Float.roundTo(aReg, 8, 23)
}

class FloatTest extends AnyFunSuite {
  test("roundTo drops a single mantissa bit (FP8_E4M3 -> FP8_E5M2)") {
    SpinalConfig().generateVerilog(FloatRoundToDrop1TestComp())
  }

  test("mul on a single-mantissa-bit format (FP4_E2M1)") {
    SpinalConfig().generateVerilog(FloatMulFp4TestComp())
  }

  test("roundTo widens the exponent without overflow (FP4 4.0 -> FP32 4.0)") {
    SimConfig.compile(FloatRoundToWidenExpTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.sign #= false
      dut.io.a.exponent #= 3 // 2^(3-1) * 1.0 = 4.0
      dut.io.a.mantissa #= 0
      dut.clockDomain.waitSampling(2)
      assert(dut.io.c.sign.toBoolean == false, "sign changed")
      assert(dut.io.c.exponent.toInt == 129, s"exponent: got ${dut.io.c.exponent.toInt}, expected 129")
      assert(dut.io.c.mantissa.toInt == 0, s"mantissa: got ${dut.io.c.mantissa.toInt}, expected 0")
    }
  }
}
