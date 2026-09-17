// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import org.scalatest.funsuite.AnyFunSuite

class MathLUTsTest extends AnyFunSuite {
  private val encFp8 = MathLUTs.floatEncodeFn(4, 3) // FP8 E4M3

  test("floatEncodeFn saturates +Infinity to max finite 448 for E4M3 (no infinity)") {
    val got = encFp8(Double.PositiveInfinity)
    assert(got == BigInt(0x7E), s"got 0x${got.toString(16)}, expected 0x7E (448)")
  }

  test("floatEncodeFn saturates -Infinity to -448 for E4M3 with sign") {
    val got = encFp8(Double.NegativeInfinity)
    assert(got == BigInt(0xFE), s"got 0x${got.toString(16)}, expected 0xFE (-448)")
  }

  test("floatEncodeFn maps NaN to zero (golden dtype convention)") {
    val got = encFp8(Double.NaN)
    assert(got == BigInt(0), s"got 0x${got.toString(16)}, expected 0")
  }

  test("floatEncodeFn saturates finite E4M3 overflow to 448 (exp 15, mant 6)") {
    val pos = encFp8(1e10)
    assert(pos == BigInt(0x7E), s"+overflow: got 0x${pos.toString(16)}, expected 0x7E (448)")
    val neg = encFp8(-1e10)
    assert(neg == BigInt(0xFE), s"-overflow: got 0x${neg.toString(16)}, expected 0xFE (-448)")
  }

  test("floatEncodeFn preserves finite E4M3 field-15 values (256..448)") {
    assert(encFp8(256.0) == BigInt(0x78), "256.0 must stay (15, 0), not saturate")
    assert(encFp8(300.0) == BigInt(0x79), "300.0 must stay (15, 1), not saturate")
    assert(encFp8(240.0) == BigInt(0x77), "240.0 must stay (14, 7)")
  }

  test("floatEncodeFn still saturates BF16 overflow to canonical infinity") {
    val encBf16 = MathLUTs.floatEncodeFn(8, 7)
    assert(encBf16(1e40) == BigInt(0x7F80), "BF16 +overflow must stay canonical infinity")
  }
}
