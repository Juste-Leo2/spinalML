package spinalML.replica

import scala.collection.mutable.ArrayBuffer

/**
 * Universal bit-exact software replica of the hardware arithmetic in SpinalML.
 * 
 * Replicates the exact hardware conventions:
 *  - Flush subnormals to +0 on underflow
 *  - Round-to-nearest-even (RN-even) for all floating-point multiplications and additions
 *  - Overflow saturates to infinity (exponent all ones)
 *  - Logarithmic adder trees matching the RTL datapath accumulation
 */
object HWArithmetic {

  final case class F(s: Boolean, e: Int, m: Int)

  val PZERO = F(false, 0, 0)

  /** Decodes a finite field triple to its real value (+0 for the zero class). */
  def decode(f: F, expBits: Int, mantBits: Int): Double = {
    if (f.e == 0) return 0.0
    val bias = (1 << (expBits - 1)) - 1
    val mag = (1.0 + f.m.toDouble / (1 << mantBits)) * math.pow(2, f.e - bias)
    if (f.s) -mag else mag
  }

  /**
   * Hardware multiplication: RN-even rounding, subnormal flush to 0, overflow to Inf.
   */
  def fmul(a: F, b: F, expBits: Int, mantBits: Int): F = {
    val bias = (1 << (expBits - 1)) - 1
    val sign = a.s != b.s
    val aZero = a.e == 0
    val bZero = b.e == 0
    val mA = if (aZero) BigInt(0) else BigInt((1 << mantBits) | a.m)
    val mB = if (bZero) BigInt(0) else BigInt((1 << mantBits) | b.m)
    val prod = mA * mB

    val overflow = prod.testBit(2 * mantBits + 1)
    val normMant =
      if (overflow) (prod >> (mantBits + 1)) & ((BigInt(1) << (mantBits + 1)) - 1)
      else (prod >> mantBits) & ((BigInt(1) << (mantBits + 1)) - 1)
    val guard = if (overflow) prod.testBit(mantBits) else prod.testBit(mantBits - 1)
    val sticky =
      if (overflow) (prod & ((BigInt(1) << mantBits) - 1)) != 0
      else (prod & ((BigInt(1) << (mantBits - 1)) - 1)) != 0
    val roundUp = guard && (sticky || normMant.testBit(0))
    val rounded = normMant + (if (roundUp) BigInt(1) else BigInt(0))
    val mantOv = rounded.testBit(mantBits + 1)
    val finalMant = if (mantOv) 0 else (rounded & ((BigInt(1) << mantBits) - 1)).toInt

    val expSum = a.e + b.e - bias +
      (if (overflow) 1 else 0) + (if (mantOv) 1 else 0)

    if (aZero || bZero || expSum <= 0) PZERO
    else if (expSum >= (1 << expBits) - 1) F(sign, (1 << expBits) - 1, 0)
    else F(sign, expSum, finalMant)
  }

  /**
   * Hardware addition: magnitude alignment with 3 guard bits, RN-even normalization.
   */
  def fadd(a: F, b: F, expBits: Int, mantBits: Int): F = {
    val aZero = a.e == 0
    val bZero = b.e == 0
    val magAge = (a.e > b.e) || (a.e == b.e && a.m >= b.m)
    val larger = if (magAge) a else b
    val smaller = if (magAge) b else a
    val lZero = larger.e == 0
    val sZero = smaller.e == 0

    val lmExt = (if (lZero) BigInt(0) else BigInt((1 << mantBits) | larger.m)) << 3
    val smExt = (if (sZero) BigInt(0) else BigInt((1 << mantBits) | smaller.m)) << 3

    val expDiff = larger.e - smaller.e
    val shiftAmount = math.max(0, math.min(expDiff, mantBits + 5))
    val smShifted = smExt >> shiftAmount

    val sameSign = larger.s == smaller.s
    val mantSumExt = if (sameSign) lmExt + smShifted else lmExt - smShifted

    if (aZero && bZero) return PZERO
    if (mantSumExt == 0) return PZERO

    val bitLen = mantSumExt.bitLength
    val lz = (mantBits + 5) - bitLen
    val finalMantissa = (mantSumExt >> (bitLen - 1 - mantBits)) & ((BigInt(1) << mantBits) - 1)
    val guardPos = bitLen - 2 - mantBits
    val guardA = if (guardPos >= 0) mantSumExt.testBit(guardPos) else false
    val stickyMask = (BigInt(1) << math.max(guardPos, 0)) - 1
    val stickyA = if (guardPos > 0) (mantSumExt & stickyMask) != 0 else false
    val roundUpA = guardA && (stickyA || finalMantissa.testBit(0))
    val rounded = finalMantissa + (if (roundUpA) BigInt(1) else BigInt(0))
    val mantOvA = rounded.testBit(mantBits)
    val finalMantA = if (mantOvA) 0 else (rounded & ((BigInt(1) << mantBits) - 1)).toInt

    val newExp = larger.e + 1 - lz + (if (mantOvA) 1 else 0)

    if (newExp <= 0) PZERO
    else if (newExp >= (1 << expBits) - 1) F(larger.s, (1 << expBits) - 1, 0)
    else F(larger.s, newExp, finalMantA)
  }

  def gt(a: F, b: F): Boolean = {
    val aZero = a.e == 0
    val bZero = b.e == 0
    if (aZero && bZero) false
    else if (aZero) b.s
    else if (bZero) !a.s
    else if (a.s != b.s) !a.s && b.s
    else if (!a.s) (a.e > b.e) || (a.e == b.e && a.m > b.m)
    else (a.e < b.e) || (a.e == b.e && a.m < b.m)
  }

  def fmax(a: F, b: F, expBits: Int, mantBits: Int): F = if (gt(a, b)) a else b

  /** Mantissa truncation from integer (mirroring spinalML.utils.Float.fromSInt). */
  def fromSInt(v: Long, w: Int, expBits: Int, mantBits: Int): F = {
    val neg = v < 0
    val abs = math.abs(v) & ((1L << w) - 1)
    if (abs == 0) return PZERO
    val p = 63 - java.lang.Long.numberOfLeadingZeros(abs)
    val bias = (1 << (expBits - 1)) - 1
    val expS = bias + p
    if (expS >= (1 << expBits) - 1) return F(neg, (1 << expBits) - 1, 0)
    val padding = math.max(0, mantBits + 1 - w)
    val wp = w + padding
    val padded = abs << ((w - 1 - p) + padding)
    val mant = ((padded >> (wp - 1 - mantBits)) & ((1 << mantBits) - 1)).toInt
    F(neg, expS, mant)
  }

  def fromDouble(value: Double, expBits: Int, mantBits: Int): F = {
    val bias = (1 << (expBits - 1)) - 1
    if (value == 0.0 || value.isNaN) return PZERO
    val signBit = value < 0
    val absVal = math.abs(value)
    val maxExp = (1 << expBits) - 2
    val maxMant = (1 << mantBits) - 1
    val maxVal = (1.0 + maxMant.toDouble / (1 << mantBits)) * math.pow(2, maxExp - bias)
    if (value.isInfinity || absVal > maxVal) return F(signBit, (1 << expBits) - 1, 0)

    val e = Math.getExponent(absVal)
    val m = java.lang.Math.scalb(absVal, -e)
    var expVal = e + bias

    val scaled = BigDecimal((m - 1.0) * (1 << mantBits).toDouble)
    var mantVal = scaled.setScale(0, BigDecimal.RoundingMode.HALF_EVEN).toLongExact

    if (mantVal >= (1 << mantBits)) {
      mantVal = 0
      expVal += 1
    }

    if (expVal >= ((1 << expBits) - 1)) {
      expVal = (1 << expBits) - 1
      mantVal = 0
    } else if (expVal <= 0) {
      expVal = 0
      mantVal = 0
    }

    F(signBit, expVal.toInt, mantVal.toInt)
  }

  /**
   * Hardware logarithmic adder tree.
   * Consecutive pairs summed left-to-right at each level, odd node passed through.
   */
  def tree(xs: Seq[F], expBits: Int, mantBits: Int): F = {
    var cur = xs
    while (cur.length > 1) {
      val next = ArrayBuffer[F]()
      var i = 0
      while (i < cur.length) {
        if (i + 1 < cur.length) next += fadd(cur(i), cur(i + 1), expBits, mantBits)
        else next += cur(i)
        i += 2
      }
      cur = next.toSeq
    }
    if (cur.isEmpty) PZERO else cur.head
  }

  // Common precision encoders
  def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  def fp8Byte(f: Float): Int = {
    if (f == 0f) return 0
    val sign = if (f < 0) 0x80 else 0
    val a = math.abs(f.toDouble)
    if (a >= math.pow(2, -6)) {
      var e = math.floor(math.log(a) / math.log(2)).toInt
      var mF = a / math.pow(2, e)
      if (mF >= 2) { mF /= 2; e += 1 }
      var m = math.floor((mF - 1) * 8 + 0.5).toInt
      if (m == 8) { m = 0; e += 1 }
      sign | ((e + 7) << 3) | m
    } else {
      sign | math.floor(a * 512 + 0.5).toInt
    }
  }

  def fp8FieldsOfByte(b: Int): F = F((b >> 7 & 1) == 1, (b >> 3) & 0xF, b & 7)
}
