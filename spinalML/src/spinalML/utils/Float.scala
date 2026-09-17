// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import spinal.core._
import spinalML.dtypes.FloatML

object Float {

  /**
   * E4M3 (`fn`) is the only format in this project without infinity
   * (IEEE-style bias 7: finite values use the exponent field 15 up to
   * 448, mantissa 7 at field 15 is the single NaN).
   */
  def isE4M3(expBits: Int, mantBits: Int): Boolean = expBits == 4 && mantBits == 3

  /**
   * Saturation encoding for float overflow (Wave 4 DTYPE-07, compile-time).
   *
   * E4M3 saturates to its max finite value 448 (exponent field 15, mantissa
   * 6), per the `float8_e4m3fn` convention (no infinity). Every other format
   * saturates to canonical infinity (exponent all-ones, mantissa zero).
   *
   * NaN (`e4m3fn` mantissa 7 at field 15, decoded 480 by the goldens) is
   * never emitted here: full NaN propagation is Wave 5
   * (see `docs/rounding_policy.md` §5).
   */
  def satEncoding(expBits: Int, mantBits: Int): (Int, Int) = {
    if (isE4M3(expBits, mantBits)) (15, 6)
    else ((1 << expBits) - 1, 0)
  }

  /**
   * Saturation predicate for float overflow (Wave 4 DTYPE-07, elaborated).
   *
   * Returns True when the computed exponent/mantissa pair is at or past the
   * last finite value. E4M3 keeps finite values on the field 15
   * (256..448) and only saturates past it, or on the NaN slot
   * (mantissa all-ones at field 15). Other formats saturate on any
   * all-ones exponent field.
   */
  def saturates(expBits: Int, mantBits: Int, expSInt: SInt, mantissa: UInt): Bool = {
    val expMax = (1 << expBits) - 1
    val mantMax = (1 << mantBits) - 1
    if (isE4M3(expBits, mantBits))
      (expSInt > expMax) || (expSInt === expMax && mantissa === mantMax)
    else
      expSInt >= expMax
  }

  /**
   * Hardware combinatorial circuit to multiply two FloatML types.
   * This logic will be synthesized into DSP blocks and LUTs.
   */
  def mul(a: FloatML, b: FloatML): FloatML = {
    require(a.expBits == b.expBits && a.mantBits == b.mantBits, "Floats must be of same format to multiply")
    
    val expBits = a.expBits
    val mantBits = a.mantBits
    val bias = a.bias
    
    val c = FloatML(expBits, mantBits)
    
    // 1. Sign XOR
    c.sign := a.sign ^ b.sign
    
    // 2. Handle Zero
    val a_is_zero = a.exponent === 0
    val b_is_zero = b.exponent === 0
    
    // Add implicit leading '1' to mantissa (if not zero)
    val mantA = UInt((mantBits + 1) bits).dontSimplifyIt()
    mantA := Mux(a_is_zero, U(0, (mantBits + 1) bits), (B"1" ## a.mantissa).asUInt)
    val mantB = UInt((mantBits + 1) bits).dontSimplifyIt()
    mantB := Mux(b_is_zero, U(0, (mantBits + 1) bits), (B"1" ## b.mantissa).asUInt)
    
    // Multiply mantissas (This naturally maps to DSP blocks)
    // Result width: (mantBits + 1) * 2
    val mantProd = mantA * mantB
    
    // If MSB is 1, the product overflowed (e.g. 1.x * 1.y = 10.z) and needs a 1-bit right shift
    val overflow = mantProd.msb

    // 3. Renormalize Mantissa (Round-to-nearest-even on the dropped bits)
    val normMantProd = Mux(overflow,
      mantProd(2 * mantBits downto mantBits + 1),
      mantProd(2 * mantBits - 1 downto mantBits)
    )
    val guardM = Mux(overflow, mantProd(mantBits), mantProd(mantBits - 1))
    val stickyM = Mux(overflow,
      (mantProd(mantBits - 1 downto 0) =/= 0),
      (mantProd(mantBits - 2 downto 0) =/= 0)
    )
    val roundUpM = guardM && (stickyM || normMantProd.lsb)
    val mantRoundedExt = normMantProd +^ roundUpM.asUInt
    val mantOvM = mantRoundedExt.msb
    val finalMantM = Mux(mantOvM, U(0, mantBits bits), mantRoundedExt(mantBits - 1 downto 0))

    // 4. Exponent Addition
    // Widened so the sum can never wrap before the saturation check
    // (e.g. FP4: 2+2-1+1 = 4 overflows a 3-bit signed accumulator)
    val expSumWidth = expBits + 3
    val expSumSInt = a.exponent.intoSInt.resize(expSumWidth) +
      b.exponent.intoSInt.resize(expSumWidth) -
      bias +
      overflow.asUInt.intoSInt.resized +
      mantOvM.asUInt.intoSInt.resized                 // rounding carry adjusts the exponent

    // 5. Overflow / Underflow Checks and Final Assignment
    val (satExpM, satMantM) = satEncoding(expBits, mantBits)
    when(a_is_zero || b_is_zero || expSumSInt <= 0) {
      // Underflow or Zero
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(saturates(expBits, mantBits, expSumSInt, finalMantM)) {
      // Overflow (saturate: 448 for E4M3, infinity otherwise)
      c.exponent := satExpM
      c.mantissa := satMantM
    } otherwise {
      // Normal range
      c.exponent := expSumSInt.asUInt.resized
      c.mantissa := finalMantM
    }
    
    c
  }

  /**
   * Generates a zero float.
   */
  def zero(expBits: Int, mantBits: Int): FloatML = {
    val z = FloatML(expBits, mantBits)
    z.sign := False
    z.exponent := 0
    z.mantissa := 0
    z
  }

  /**
   * Greater-Than comparison for FloatML.
   */
  def gt(a: FloatML, b: FloatML): Bool = {
    require(a.expBits == b.expBits && a.mantBits == b.mantBits)
    
    val signGt = (a.sign === False) && (b.sign === True)
    val signEq = a.sign === b.sign
    
    val magGt = (a.exponent > b.exponent) || ((a.exponent === b.exponent) && (a.mantissa > b.mantissa))
    val negMagGt = (a.exponent < b.exponent) || ((a.exponent === b.exponent) && (a.mantissa < b.mantissa))
    
    val a_zero = a.exponent === 0
    val b_zero = b.exponent === 0
    
    val res = Bool()
    when(a_zero && b_zero) {
      res := False
    } elsewhen(a_zero) {
      res := (b.sign === True)
    } elsewhen(b_zero) {
      res := (a.sign === False)
    } otherwise {
      res := signGt || (signEq && Mux(a.sign, negMagGt, magGt))
    }
    res
  }

  /**
   * Returns the maximum of two FloatML values.
   */
  def max(a: FloatML, b: FloatML): FloatML = Mux(gt(a, b), a, b)

  /**
   * Hardware combinatorial circuit to add/subtract two FloatML types.
   */
  def add(a: FloatML, b: FloatML): FloatML = {
    require(a.expBits == b.expBits && a.mantBits == b.mantBits)
    val expBits = a.expBits
    val mantBits = a.mantBits
    val c = FloatML(expBits, mantBits)
    
    val a_zero = a.exponent === 0
    val b_zero = b.exponent === 0
    
    // 1. Sort by magnitude
    val magA_ge_magB = (a.exponent > b.exponent) || ((a.exponent === b.exponent) && (a.mantissa >= b.mantissa))
    val larger = Mux(magA_ge_magB, a, b)
    val smaller = Mux(magA_ge_magB, b, a)
    val larger_zero = larger.exponent === 0
    val smaller_zero = smaller.exponent === 0
    
    val expDiff = larger.exponent - smaller.exponent
    
    // Add implicit 1 to mantissas
    val largerMant = Mux(larger_zero, U(0, (mantBits + 1) bits), (B"1" ## larger.mantissa).asUInt)
    val smallerMant = Mux(smaller_zero, U(0, (mantBits + 1) bits), (B"1" ## smaller.mantissa).asUInt)
    
    // 2. Shift smaller mantissa to align
    val guardBits = 3
    val largerMantExt = largerMant @@ U(0, guardBits bits)
    val smallerMantExt = smallerMant @@ U(0, guardBits bits)
    
    val maxShift = mantBits + guardBits + 2
    val shiftWidth = Math.max(expDiff.getWidth, spinal.core.log2Up(maxShift + 1))
    val expDiffWider = expDiff.resize(shiftWidth)
    val shiftAmount = Mux(expDiffWider > maxShift, U(maxShift, shiftWidth bits), expDiffWider)
    val smallerMantShifted = smallerMantExt >> shiftAmount
    
    // 3. Add or Subtract
    val sameSign = larger.sign === smaller.sign
    val subRes = largerMantExt - smallerMantShifted
    val mantSumExt = Mux(sameSign,
      largerMantExt +^ smallerMantShifted,  
      subRes.resize(subRes.getWidth + 1)
    )
    
    val W = mantBits + guardBits + 2 // Total width of mantSumExt
    
    // 4. Renormalize (Leading Zero Detection)
    val reversed = mantSumExt.asBits.reversed
    val lz = spinal.lib.OHToUInt(spinal.lib.OHMasking.first(reversed))
    
    val normalizedSumExt = mantSumExt << lz
    val finalMantissa = normalizedSumExt(W - 2 downto W - 1 - mantBits)
    
    // Round-to-nearest-even on the dropped bits:
    // guard bit sits just below the mantissa window, sticky is the OR of the rest
    val guardA = normalizedSumExt(W - 2 - mantBits)
    val stickyA = (normalizedSumExt(W - 3 - mantBits downto 0) =/= 0)
    val roundUpA = guardA && (stickyA || finalMantissa.lsb)
    val mantRoundedExt = finalMantissa +^ roundUpA.asUInt
    val mantOvA = mantRoundedExt.msb
    val finalMantA = Mux(mantOvA, U(0, mantBits bits), mantRoundedExt(mantBits - 1 downto 0))
    
    val expAdjustSInt = 1 - lz.intoSInt
    // Widened so exponent+1 can never wrap before the saturation check
    val newExpSInt = larger.exponent.intoSInt.resize(expBits + 3) +
      expAdjustSInt.resize(expBits + 3) +
      mantOvA.asUInt.intoSInt.resized               // rounding carry adjusts the exponent
    
    // 5. Pack result
    c.sign := larger.sign
    val sumIsZero = mantSumExt === 0
    val (satExpA, satMantA) = satEncoding(expBits, mantBits)

    when(a_zero && b_zero) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(sumIsZero || newExpSInt <= 0) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(saturates(expBits, mantBits, newExpSInt, finalMantA)) {
      c.exponent := satExpA
      c.mantissa := satMantA
    } otherwise {
      c.exponent := newExpSInt.asUInt.resized
      c.mantissa := finalMantA
    }
    
    c
  }

  /**
   * Pure elaboration-time logic converting a Double into FloatML fields
   * (sign, biased exponent, mantissa). Mirrors the Python golden model
   * `FloatML.from_float` bit-exactly (banker's rounding on the mantissa,
   * overflow -> saturation encoding, underflow -> zero).
   */
  def doubleToFields(value: Double, expBits: Int, mantBits: Int): (Boolean, Int, Long) = {
    val bias = (1 << (expBits - 1)) - 1
    val (satExp, satMant) = satEncoding(expBits, mantBits)

    if (value == 0.0 || value.isNaN) {
      return (false, 0, 0)
    }

    val signBit = value < 0
    val absVal = math.abs(value)

    // Saturation check (same formula as the golden model). E4M3 tops out at
    // the max finite 448 (field 15, mantissa 6); the mantissa-7 slot is NaN.
    val (maxExp, maxMant, maxVal) = if (isE4M3(expBits, mantBits)) {
      (15, 6, 448.0)
    } else {
      val me = (1 << expBits) - 2
      val mm = (1 << mantBits) - 1
      (me, mm, (1.0 + mm.toDouble / (1 << mantBits)) * math.pow(2, me - bias))
    }

    if (value.isInfinity || absVal > maxVal) {
      return (signBit, satExp, satMant)
    }

    // frexp equivalent: m in [1, 2), e = floor(log2(absVal))
    val e = Math.getExponent(absVal)
    val m = java.lang.Math.scalb(absVal, -e)
    var expVal = e + bias

    // Mantissa rounding: Python round() == half-to-even
    val scaled = BigDecimal((m - 1.0) * (1 << mantBits).toDouble)
    var mantVal = scaled.setScale(0, BigDecimal.RoundingMode.HALF_EVEN).toLongExact

    // Mantissa rounding overflow FIRST (can rescue an underflow)
    if (mantVal >= (1 << mantBits)) {
      mantVal = 0
      expVal += 1
    }

    // THEN saturation / underflow (subnormals are omitted in hardware).
    // E4M3 keeps finite field-15 values (256..448); only past-the-max or
    // the NaN slot (mantissa 7 at field 15) saturates to 448.
    val needsSat = if (isE4M3(expBits, mantBits))
      expVal > 15 || (expVal == 15 && mantVal == 7)
    else
      expVal >= ((1 << expBits) - 1)
    if (needsSat) {
      expVal = satExp
      mantVal = satMant
    } else if (expVal <= 0) {
      expVal = 0
      mantVal = 0
    }

    (signBit, expVal, mantVal)
  }

  /**
   * Elaboration-time constant conversion: Double -> FloatML hardware literal.
   * Bit-exact counterpart of the Python golden model `FloatML.from_float`.
   */
  def fromDouble(value: Double, expBits: Int, mantBits: Int): FloatML = {
    val (signBit, expVal, mantVal) = doubleToFields(value, expBits, mantBits)
    val c = FloatML(expBits, mantBits)
    c.sign := Bool(signBit)
    c.exponent := U(expVal, expBits bits)
    c.mantissa := U(mantVal, mantBits bits)
    c
  }

  /**
   * Hardware circuit to convert an SInt into a FloatML.
   */
  def fromSInt(inValue: SInt, expBits: Int, mantBits: Int): FloatML = {
    val W = inValue.getBitsWidth
    val c = FloatML(expBits, mantBits)
    
    // 1. Sign
    c.sign := inValue < 0
    
    // 2. Absolute value
    val absVal = inValue.abs
    
    // 3. Find leading zero (LZD)
    val isZero = absVal === 0
    val reversed = absVal.asBits.reversed
    val lz = spinal.lib.OHToUInt(spinal.lib.OHMasking.first(reversed))
    
    val posSInt = S(W - 1, lz.getWidth + 2 bits) - lz.intoSInt
    val expSInt = S(c.bias, expBits + 2 bits) + posSInt
    
    // 4. Align mantissa
    val absValShiftedLeft = absVal << lz
    
    val paddingBits = math.max(0, mantBits + 1 - W)
    val paddedVal = if(paddingBits > 0) (absValShiftedLeft @@ U(0, paddingBits bits)) else absValShiftedLeft
    
    val W_padded = W + paddingBits
    val mantissa = paddedVal(W_padded - 2 downto W_padded - 1 - mantBits)
    
    // 5. Final assignment
    val (satExpS, satMantS) = satEncoding(expBits, mantBits)
    when(isZero) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(saturates(expBits, mantBits, expSInt, mantissa)) {
      c.exponent := satExpS
      c.mantissa := satMantS
    } otherwise {
      c.exponent := expSInt.asUInt.resized
      c.mantissa := mantissa
    }
    
    c
  }

  /**
   * Negate a FloatML value (sign flip). Zero preserved as positive.
   */
  def neg(a: FloatML): FloatML = {
    val c = FloatML(a.expBits, a.mantBits)
    c.sign := a.sign && (a.exponent =/= 0)
    c.exponent := a.exponent
    c.mantissa := a.mantissa
    c
  }

  /**
   * Exact runtime up-widening of a FloatML into a wider format
   * (mantBits can only grow; exponent bias is re-based). NaN/Inf payloads
   * are preserved: exponent 0 stays zero, exponent all-ones stays all-ones.
   */
  def widen(a: FloatML, outExpBits: Int, outMantBits: Int): FloatML = {
    require(outMantBits >= a.mantBits, "widen requires outMantBits >= in.mantBits")
    require(outExpBits >= a.expBits, "widen requires outExpBits >= in.expBits")
    val c = FloatML(outExpBits, outMantBits)
    val biasDelta = ((1 << (outExpBits - 1)) - 1) - a.bias

    c.sign := a.sign
    // The fraction (1.m) stays normalized in [1, 2): widen it by
    // left-justifying, NOT by LSB-resize (which would place m at the least
    // significant bits and collapse 1.125 => 1.0 + 2^-23).
    c.mantissa := (a.mantissa << (outMantBits - a.mantBits)).resize(outMantBits)

    val expSInt = a.exponent.intoSInt.resize(outExpBits + 2 bits) + biasDelta
    when(a.exponent === 0) {
      c.exponent := 0
    } elsewhen(expSInt >= ((1 << outExpBits) - 1)) {
      c.exponent := ((1 << outExpBits) - 1)
      c.mantissa := 0
    } otherwise {
      c.exponent := expSInt.asUInt.resized
    }
    c
  }

  /**
   * Runtime round-to-nearest-even of a FloatML into a narrower format
   * (expBits not necessarily smaller, mantBits can shrink). Mirrors the
   * golden model's dtype.from_float rounding: mantissa overflow carries
   * into the exponent, underflow yields zero, overflow saturates to the
   * saturation encoding (448 for E4M3, inf-encoding otherwise).
   */
  def roundTo(a: FloatML, outExpBits: Int, outMantBits: Int): FloatML = {
    val c = FloatML(outExpBits, outMantBits)
    val biasDelta = ((1 << (outExpBits - 1)) - 1) - a.bias
    // The exponent sum must hold the input exponent plus the bias delta (which
    // can be large when widening towards a bigger exponent bias, e.g. FP4 -> FP32)
    val expSIntWidth = (a.expBits max outExpBits) + 4
    val (satExpR, satMantR) = satEncoding(outExpBits, outMantBits)

    val aZero = a.exponent === 0 && a.mantissa === 0

    when(aZero) {
      c.sign := False
      c.exponent := 0
      c.mantissa := 0
    } otherwise {
      c.sign := a.sign
      if (a.mantBits > outMantBits) {
        val drop = a.mantBits - outMantBits
        val mantExt = (B"1" ## a.mantissa).asUInt           // hidden 1 + mantBits
        val kept = mantExt(a.mantBits downto drop)          // outMantBits + 1 bits
        val guard = mantExt(drop - 1)
        val sticky = mantExt(drop - 2 downto 0) =/= 0

        val roundUp = guard && (sticky || kept.lsb)
        val mantRnd = kept +^ roundUp.asUInt
        val mantOv = mantRnd.msb

        val expSInt = a.exponent.intoSInt.resize(expSIntWidth bits) +
          biasDelta +
          mantOv.asUInt.intoSInt.resized

        when(saturates(outExpBits, outMantBits, expSInt,
            Mux(mantOv, U(0, outMantBits bits), mantRnd(outMantBits - 1 downto 0)))) {
          c.exponent := satExpR
          c.mantissa := satMantR
        } elsewhen(expSInt <= 0) {
          c.exponent := 0
          c.mantissa := 0
          c.sign := False
        } otherwise {
          c.exponent := expSInt.asUInt.resized
          c.mantissa := Mux(mantOv, U(0, outMantBits bits), mantRnd(outMantBits - 1 downto 0))
        }
      } else  {
        // Exact widening path: fraction stays normalized, left-justified.
        c.mantissa := (a.mantissa << (outMantBits - a.mantBits)).resize(outMantBits)
        val expSInt = a.exponent.intoSInt.resize(expSIntWidth bits) + biasDelta
        when(saturates(outExpBits, outMantBits, expSInt, c.mantissa)) {
          c.exponent := satExpR
          c.mantissa := satMantR
        } elsewhen(expSInt <= 0) {
          c.exponent := 0
          c.mantissa := 0
          c.sign := False
        } otherwise {
          c.exponent := expSInt.asUInt.resized
        }
      }
    }
    c
  }
}
