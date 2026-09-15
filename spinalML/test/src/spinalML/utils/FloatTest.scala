// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
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

class FloatTest extends AnyFunSuite {
  test("roundTo drops a single mantissa bit (FP8_E4M3 -> FP8_E5M2)") {
    SpinalConfig().generateVerilog(FloatRoundToDrop1TestComp())
  }

  test("mul on a single-mantissa-bit format (FP4_E2M1)") {
    SpinalConfig().generateVerilog(FloatMulFp4TestComp())
  }
}
