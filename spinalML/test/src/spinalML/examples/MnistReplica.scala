// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import scala.collection.mutable.ArrayBuffer

/**
 * Software bit-exact replica of the FloatML hardware arithmetic
 * ([[spinalML.utils.Float]]), used by the Mnist benches as an oracle for
 * arbitrary (random) inputs: every logit produced by the simulated datapath
 * must equal the replica's prediction exactly.
 *
 * Values are (sign, biasedExponent, mantissa) field triples. Key hardware
 * conventions replicated here:
 *  - NO subnormals are ever computed: underflow flushes to +0;
 *  - an operand whose exponent field is ZERO behaves as zero regardless of
 *    its mantissa field — this is how subnormal-ENCODED DDR constants
 *    (weights/bias bytes) behave once they reach the datapath;
 *  - rounding is round-to-nearest-even everywhere (mul, add);
 *  - overflow saturates to the infinity encoding (exponent all ones);
 *  - `fromSInt` TRUNCATES the mantissa (no rounding), mirroring the RTL.
 */
object HWFloat {

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
   * Port of `spinalML.utils.Float.mul`: full mantissa product, RN-even
   * rounding, underflow -> +0, overflow -> Inf, zero class on exponent == 0.
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
   * Port of `spinalML.utils.Float.add`: magnitude-sorted alignment (field
   * comparison, 3 guard bits, capped shift), RN-even normalization,
   * underflow -> +0, overflow -> Inf carrying the LARGER operand's sign.
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
    val lz = (mantBits + 5) - bitLen // leading zeros within the W-bit vector
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

  /**
   * Port of `spinalML.utils.Float.gt` (zero class included: two zero-field
   * operands compare False, so `max(a, b)` then returns b verbatim).
   */
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

  /**
   * Port of `spinalML.utils.Float.fromSInt`: mantissa TRUNCATION (no
   * rounding), saturation keeps the input sign.
   */
  def fromSInt(v: Long, w: Int, expBits: Int, mantBits: Int): F = {
    val neg = v < 0
    val abs = math.abs(v) & ((1L << w) - 1)
    if (abs == 0) return PZERO
    val p = 63 - java.lang.Long.numberOfLeadingZeros(abs)
    val bias = (1 << (expBits - 1)) - 1
    val expS = bias + p
    if (expS >= (1 << expBits) - 1) return F(neg, (1 << expBits) - 1, 0)
    // Align the MSB to bit (w-1) exactly like the RTL's `absVal << lz`,
    // then append sub-mantissa zero padding for narrow targets.
    val padding = math.max(0, mantBits + 1 - w)
    val wp = w + padding
    val padded = abs << ((w - 1 - p) + padding)
    val mant = ((padded >> (wp - 1 - mantBits)) & ((1 << mantBits) - 1)).toInt
    F(neg, expS, mant)
  }

  /**
   * Verbatim port of `spinalML.utils.Float.doubleToFields`: banker's
   * rounding on the mantissa, overflow -> infinity, underflow -> zero
   * (no subnormals).
   */
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
   * Port of MatmulOp's logarithmic adder tree association: consecutive
   * pairs summed left-to-right at each level, odd node passed through.
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
}

/**
 * Bit-exact software replica of the [[Mnist]] BF16 forward pass
 * (Conv 5x5 -> ReLU -> MaxPool 2x2 -> Flatten -> Linear 288->10, all BF16).
 *
 * The DDR-side encoders (`bf16Bits`, `fp8Bytes`) are kept in deliberate
 * lockstep with the bench packers: the replica consumes the SAME encoded
 * bytes the DMA streams into the datapath, so encoded-but-subnormal
 * constants follow the hardware zero-class rule instead of their real value.
 */
object MnistReplica {
  import HWFloat._

  val EB = 8 // BF16 exponent bits
  val MB = 7 // BF16 mantissa bits

  def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  private def kernelFields(kernel: Seq[Float]): Seq[F] = kernel.map(bf16Fields)

  /** Full-network logits (decoded doubles) for one binarized image. */
  def logits(img: Seq[String]): Seq[Double] = logitsK(img, 288)

  /** Same oracle with the M2 K-chunk width of the Linear (wLanes). */
  def logitsK(img: Seq[String], wLanes: Int): Seq[Double] = {
    val pix = Array.ofDim[Int](28, 28)
    for (y <- 0 until 28; x <- 0 until 28) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    // Conv 5x5 valid + bias, then ReLU — everything in rounded BF16
    val convOut = Array.ofDim[F](2, 24, 24)
    val kernels = MnistWeights.convW.map(kernelFields)
    val biases = MnistWeights.convB.map(bf16Fields)
    for (c <- 0 until 2; wy <- 0 until 24; wx <- 0 until 24) {
      val prods = for (r <- 0 until 5; k <- 0 until 5)
        yield fmul(bf16Fields(pix(wy + r)(wx + k).toFloat), kernels(c)(r * 5 + k), EB, MB)
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      val biased = fadd(acc, biases(c), EB, MB)
      convOut(c)(wy)(wx) = if (biased.s) PZERO else biased // ReLU: sign -> +0, like the RTL
    }

    // MaxPool 2x2 stride 2, features-last flatten order (i, j, c)
    val acts = ArrayBuffer[F]()
    for (i <- 0 until 12; j <- 0 until 12; c <- 0 until 2) {
      val v = fmax(fmax(convOut(c)(2 * i)(2 * j), convOut(c)(2 * i)(2 * j + 1), EB, MB),
        fmax(convOut(c)(2 * i + 1)(2 * j), convOut(c)(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      acts += v
    }

    linearLayer(acts, MnistWeights.fcW.map(row => row.map(bf16Fields)), MnistWeights.fcB.map(bf16Fields), EB, MB, wLanes)
  }

  /** Shared tail: logits[o] = (+0 + tree(act . w[o])) + b[o], RN-rounded.
    *
    * M2: the hardware matmul splits the K axis into chunks of `wLanes`
    * lanes, accumulating `fadd(acc, tree(chunk))` per chunk in order — a
    * full-width single tree when wLanes == K (legacy, byte-identical). */
  def linearLayer(acts: Seq[F], w: Seq[Seq[F]], b: Seq[F], expBits: Int, mantBits: Int,
                  wLanes: Int = 288): Seq[Double] = {
    require(wLanes > 0 && acts.length % wLanes == 0,
      s"wLanes=$wLanes must divide the K dimension ${acts.length}")
    val chunks = acts.length / wLanes
    val out = ArrayBuffer[Double]()
    for (o <- w.indices) {
      var acc = PZERO
      for (c <- 0 until chunks) {
        val prods = for (k <- c * wLanes until (c + 1) * wLanes)
          yield fmul(acts(k), w(o)(k), expBits, mantBits)
        acc = fadd(acc, tree(prods, expBits, mantBits), expBits, mantBits)
      }
      out += decode(fadd(acc, b(o), expBits, mantBits), expBits, mantBits)
    }
    out.toSeq
  }
}

/** Bit-exact replica of the [[Mnistw4a8]] mixed-precision forward pass
  * (INT4 conv in the integer domain -> dequantizing Cast -> FP8 Linear). */
object Mnistw4a8Replica {
  import HWFloat._

  val EB = 4 // FP8 E4M3
  val MB = 3

  /** Same encoder as the bench packer (deliberate lockstep). */
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

  def fieldsOfByte(b: Int): F = F((b >> 7 & 1) == 1, (b >> 3) & 0xF, b & 7)

  /** Full-network logits for one binarized image. */
  def logits(img: Seq[String]): Seq[Double] = logitsK(img, 288)

  /** Same oracle with the M2 K-chunk width of the Linear (wLanes). */
  def logitsK(img: Seq[String], wLanes: Int): Seq[Double] = {
    val pix = Array.ofDim[Int](28, 28)
    for (y <- 0 until 28; x <- 0 until 28) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    // Integer-domain conv: I8 pixels x sign-extended I4 codes, I16 accumulator,
    // quantized bias folded in, ReLU applied pre-scale.
    val convOut = Array.ofDim[Int](2, 24, 24)
    for (c <- 0 until 2; wy <- 0 until 24; wx <- 0 until 24) {
      var acc = Mnistw4a8Weights.convBq(c)
      for (r <- 0 until 5; k <- 0 until 5)
        acc += pix(wy + r)(wx + k) * Mnistw4a8Weights.convWq(c)(r * 5 + k)
      convOut(c)(wy)(wx) = math.max(acc, 0)
    }

    // MaxPool 2x2 stride 2, then Cast(I16 -> E4M3) * convScale (per element)
    val scaleLit = fromDouble(Mnistw4a8Weights.convScale, EB, MB)
    val acts = ArrayBuffer[F]()
    for (i <- 0 until 12; j <- 0 until 12; c <- 0 until 2) {
      val v = math.max(math.max(convOut(c)(2 * i)(2 * j), convOut(c)(2 * i)(2 * j + 1)),
        math.max(convOut(c)(2 * i + 1)(2 * j), convOut(c)(2 * i + 1)(2 * j + 1)))
      acts += fmul(fromSInt(v.toLong, 16, EB, MB), scaleLit, EB, MB)
    }

    val w = Mnistw4a8Weights.fcW.map(row => row.map(v => fieldsOfByte(fp8Byte(v))))
    val b = Mnistw4a8Weights.fcB.map(v => fieldsOfByte(fp8Byte(v)))
    MnistReplica.linearLayer(acts, w, b, EB, MB, wLanes)
  }
}
