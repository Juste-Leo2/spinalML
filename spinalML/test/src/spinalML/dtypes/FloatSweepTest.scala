// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dtypes

import spinal.core._
import spinal.core.sim._
import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

/**
 * Probe component exposing Float.add / Float.mul of one format for exhaustive
 * simulation sweeps against an independent bit-level reference model.
 */
case class FloatSweepComp(expBits: Int, mantBits: Int) extends Component {
  val io = new Bundle {
    val a = in(FloatML(expBits, mantBits))
    val b = in(FloatML(expBits, mantBits))
    val s = out(FloatML(expBits, mantBits))
    val m = out(FloatML(expBits, mantBits))
  }
  io.s := spinalML.utils.Float.add(io.a, io.b)
  io.m := spinalML.utils.Float.mul(io.a, io.b)
}

/**
 * Independent bit-level reference for FloatML add/mul.
 *
 * Mirrors the Python golden model (tests/python/golden_models/ops.py) but is
 * written with plain unbounded Int arithmetic and explicit saturations, so it
 * CANNOT reproduce hardware width-truncation bugs by construction (unlike a
 * self-comparison formal oracle, see docs/symbolicTest.md).
 */
object FloatGolden {

  private def pack(sign: Boolean, exp: Int, mant: Int, e: Int, m: Int): Int =
    ((if (sign) 1 else 0) << (e + m)) | (exp << m) | mant

  private def unpack(bits: Int, e: Int, m: Int): (Boolean, Int, Int) =
    (((bits >> (e + m)) & 1) == 1, (bits >> m) & ((1 << e) - 1), bits & ((1 << m) - 1))

  /** Encodings whose value round-trips through the format (only these flow in real datapaths). */
  def isCanonical(bits: Int, e: Int, m: Int): Boolean = {
    val (_, exp, mant) = unpack(bits, e, m)
    if (exp == 0) bits == 0                       // zero: single encoding (no subnormals)
    else if (exp == (1 << e) - 1) mant == 0       // infinity: mantissa must be zero
    else true                                      // normals: always round-trip
  }

  def mul(aBits: Int, bBits: Int, e: Int, m: Int): Int = {
    val bias = (1 << (e - 1)) - 1
    val (sa, ea, ma) = unpack(aBits, e, m)
    val (sb, eb, mb) = unpack(bBits, e, m)

    if (ea == 0 || eb == 0) return 0              // zero operand -> zero product

    val sign = sa != sb
    val prod = ((1 << m) | ma) * ((1 << m) | mb)  // unbounded: cannot wrap
    var ovf = (prod >> (2 * m + 1)) & 1
    var normMant = if (ovf != 0) (prod >> (m + 1)) & ((1 << m) - 1)
                   else (prod >> m) & ((1 << m) - 1)
    val droppedM = if (ovf != 0) prod & ((1 << (m + 1)) - 1)
                   else prod & ((1 << m) - 1)
    val guardM = if (ovf != 0) (droppedM >> m) & 1
                 else (droppedM >> (m - 1)) & 1
    val stickyM = if (ovf != 0) (droppedM & ((1 << m) - 1)) != 0
                  else (droppedM & ((1 << (m - 1)) - 1)) != 0

    // Round-to-nearest-even (mirrors the hardware)
    if (guardM != 0 && (stickyM || (normMant & 1) != 0)) {
      normMant += 1
      if (normMant >= (1 << m)) { normMant = 0; ovf = 1 } // carry adjusts the exponent
    }

    val expSum = ea + eb - bias + ovf             // unbounded: cannot wrap

    if (expSum <= 0) pack(false, 0, 0, e, m)      // underflow -> zero
    else if (expSum >= ((1 << e) - 1)) pack(sign, (1 << e) - 1, 0, e, m) // saturate
    else pack(sign, expSum, normMant, e, m)
  }

  def add(aBits: Int, bBits: Int, e: Int, m: Int): Int = {
    val (sa, ea, ma) = unpack(aBits, e, m)
    val (sb, eb, mb) = unpack(bBits, e, m)
    val aZero = ea == 0
    val bZero = eb == 0

    val magAgeB = (ea > eb) || (ea == eb && ma >= mb)
    val (lSign, lExp, lMant, lZero) = if (magAgeB) (sa, ea, ma, aZero) else (sb, eb, mb, bZero)
    val (sSign, _, sMant, sZero) = if (magAgeB) (sb, eb, mb, bZero) else (sa, ea, ma, aZero)

    val expDiff = lExp - (if (magAgeB) eb else ea)
    val lFull = if (lZero) 0 else (1 << m) | lMant
    val sFull = if (sZero) 0 else (1 << m) | sMant

    val guardBits = 3
    val lExt = lFull << guardBits
    val maxShift = m + guardBits + 2
    val sShifted = sFull << guardBits >> math.min(expDiff, maxShift)

    val sumIsZeroEnc = aZero && bZero
    val raw =
      if (lSign == sSign) lExt + sShifted
      else {
        val sub = lExt - sShifted
        sub & ((1 << (m + guardBits + 3)) - 1)
      }

    val w = m + guardBits + 2
    val lz = if (raw == 0) 0 else math.max(0, w - BigInt(raw).bitLength)
    val normalized = raw << lz
    var finalMant = (normalized >> (w - 1 - m)) & ((1 << m) - 1)

    // Round-to-nearest-even (mirrors the hardware)
    val dropped = normalized & ((1 << (w - 1 - m)) - 1)
    val guardA = (dropped >> (w - 2 - m)) & 1
    val stickyA = (dropped & ((1 << (w - 2 - m)) - 1)) != 0
    var lzAdj = lz
    if (guardA != 0 && (stickyA || (finalMant & 1) != 0)) {
      finalMant += 1
      if (finalMant >= (1 << m)) { finalMant = 0; lzAdj -= 1 } // carry adjusts the exponent
    }

    val newExp = lExp + 1 - lzAdj

    if (sumIsZeroEnc || raw == 0 || newExp <= 0) pack(false, 0, 0, e, m)
    else if (newExp >= ((1 << e) - 1)) pack(lSign, (1 << e) - 1, 0, e, m)
    else pack(lSign, newExp, finalMant, e, m)
  }
}

class FloatSweepTest extends AnyFunSuite {

  private def runSweep(e: Int, m: Int, pairs: Seq[(Int, Int)]): Unit = {
    val total = 1 << (1 + e + m)
    SimConfig.compile(FloatSweepComp(e, m)).doSim { dut =>
      def set(p: FloatML, bits: Int): Unit = {
        p.sign #= ((bits >> (e + m)) & 1) == 1
        p.exponent #= (bits >> m) & ((1 << e) - 1)
        p.mantissa #= bits & ((1 << m) - 1)
      }
      def get(p: FloatML): Int =
        ((if (p.sign.toBoolean) 1 else 0) << (e + m)) |
         ((p.exponent.toInt & ((1 << e) - 1)) << m) | (p.mantissa.toInt & ((1 << m) - 1))

      var checked = 0
      for ((ab, bb) <- pairs) {
        set(dut.io.a, ab)
        set(dut.io.b, bb)
        sleep(1)

        val hwAdd = get(dut.io.s)
        val hwMul = get(dut.io.m)
        val gdAdd = FloatGolden.add(ab, bb, e, m)
        val gdMul = FloatGolden.mul(ab, bb, e, m)

        assert(hwAdd == gdAdd,
          f"ADD mismatch ($e/$m): a=$ab%02x b=$bb%02x hw=$hwAdd%02x golden=$gdAdd%02x")
        assert(hwMul == gdMul,
          f"MUL mismatch ($e/$m): a=$ab%02x b=$bb%02x hw=$hwMul%02x golden=$gdMul%02x")
        checked += 1
      }
      println(f"SWEEP $checked%d pairs checked ($e exponent bits, $m mantissa bits)")
    }
  }

  private def canonicalPairs(e: Int, m: Int): Seq[(Int, Int)] = {
    val total = 1 << (1 + e + m)
    val canon = (0 until total).filter(b => FloatGolden.isCanonical(b, e, m))
    for (a <- canon; b <- canon) yield (a, b)
  }

  test("Exhaustive sweep narrow format") {
    runSweep(2, 1, canonicalPairs(2, 1))
  }

  test("Exhaustive sweep medium format") {
    runSweep(4, 3, canonicalPairs(4, 3))
  }

  test("Randomized sweep wide format") {
    // Full 16-bit space is 4G pairs; sample heavily around high exponents
    // where saturation/wrap bugs live (the Float.mul/add wrap bug class).
    val rand = new Random(42)
    val e = 8
    val m = 7
    val maxExp = (1 << e) - 1
    def randomCanonical(): Int = {
      val exp = rand.nextInt(12) match {         // 75% of draws in the top exponents
        case i if i < 9 => maxExp - rand.nextInt(9)
        case _ => 1 + rand.nextInt(math.max(1, maxExp - 2))
      }
      val mant = rand.nextInt(1 << m)
      var bits = ((rand.nextInt(2) << (e + m)) | (exp << m) | mant)
      if (!FloatGolden.isCanonical(bits, e, m)) bits &= ~((1 << m) - 1) // force mant=0 for inf-like
      bits
    }
    val pairs = Seq.fill(20000)((randomCanonical(), randomCanonical()))
    runSweep(e, m, pairs)
  }

  test("Absorption sweep wide format") {
    // Systematic near-1.0 absorption coverage: add(x, tiny) collapsing to x.
    // This corner escaped random sampling and once broke softmax normalization
    // (add(1.0, 0.0074) -> 1.0 instead of rounding up).
    val e = 8
    val m = 7
    val allCanonical = (0 until (1 << (1 + e + m))).filter(b => FloatGolden.isCanonical(b, e, m))
    val smallExps = (110 to 127).toSet
    val smalls = allCanonical.filter { b =>
      val exp = (b >> m) & ((1 << e) - 1)
      smallExps.contains(exp) || b == 0
    }
    val ones = allCanonical.filter { b =>          // values in [1.9375 .. 4.0625] +- signs
      val exp = (b >> m) & ((1 << e) - 1)
      exp == 127 || exp == 128 || exp == 129
    }
    val pairs = for (a <- ones; b <- smalls) yield (a, b)
    runSweep(e, m, pairs)
  }
}
