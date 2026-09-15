// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import org.scalatest.funsuite.AnyFunSuite

class MathLUTsTest extends AnyFunSuite {
  private val encFp8 = MathLUTs.floatEncodeFn(4, 3) // FP8 E4M3

  test("floatEncodeFn saturates +Infinity to canonical infinity (exp all ones, mant 0)") {
    val got = encFp8(Double.PositiveInfinity)
    assert(got == BigInt(0x78), s"got 0x${got.toString(16)}, expected 0x78")
  }

  test("floatEncodeFn saturates -Infinity to canonical infinity with sign") {
    val got = encFp8(Double.NegativeInfinity)
    assert(got == BigInt(0xF8), s"got 0x${got.toString(16)}, expected 0xF8")
  }

  test("floatEncodeFn maps NaN to zero (golden dtype convention)") {
    val got = encFp8(Double.NaN)
    assert(got == BigInt(0), s"got 0x${got.toString(16)}, expected 0")
  }

  test("floatEncodeFn saturates finite overflow to canonical infinity (mant 0)") {
    val pos = encFp8(1e10)
    assert(pos == BigInt(0x78), s"+overflow: got 0x${pos.toString(16)}, expected 0x78")
    val neg = encFp8(-1e10)
    assert(neg == BigInt(0xF8), s"-overflow: got 0x${neg.toString(16)}, expected 0xF8")
  }
}
