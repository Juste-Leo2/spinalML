package spinalML.utils

import spinal.core._
import spinalML.dtypes.FloatML

object Float {
  
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
    val mantA = Mux(a_is_zero, U(0, (mantBits + 1) bits), (B"1" ## a.mantissa).asUInt)
    val mantB = Mux(b_is_zero, U(0, (mantBits + 1) bits), (B"1" ## b.mantissa).asUInt)
    
    // Multiply mantissas (This naturally maps to DSP blocks)
    // Result width: (mantBits + 1) * 2
    val mantProd = mantA * mantB
    
    // If MSB is 1, the product overflowed (e.g. 1.x * 1.y = 10.z) and needs a 1-bit right shift
    val overflow = mantProd.msb
    
    // 3. Exponent Addition
    // We use SInt to safely handle intermediate negative numbers before underflow check
    val expSumSInt = a.exponent.intoSInt + b.exponent.intoSInt - bias + overflow.asUInt.intoSInt
    
    // 4. Renormalize Mantissa
    // Extract the exact mantBits after the leading 1
    val normMantProd = Mux(overflow, 
      mantProd(2 * mantBits downto mantBits + 1),
      mantProd(2 * mantBits - 1 downto mantBits)
    )
    
    // 5. Overflow / Underflow Checks and Final Assignment
    when(a_is_zero || b_is_zero || expSumSInt <= 0) {
      // Underflow or Zero
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(expSumSInt >= ((1 << expBits) - 1)) {
      // Overflow (Saturate to Infinity)
      c.exponent := ((1 << expBits) - 1)
      c.mantissa := 0
    } otherwise {
      // Normal range
      c.exponent := expSumSInt.asUInt.resized
      c.mantissa := normMantProd
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
    val shiftAmount = Mux(expDiff > maxShift, U(maxShift), expDiff)
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
    
    val expAdjustSInt = 1 - lz.intoSInt
    val newExpSInt = larger.exponent.intoSInt + expAdjustSInt
    
    // 5. Pack result
    c.sign := larger.sign
    val sumIsZero = mantSumExt === 0
    
    when(a_zero && b_zero) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(sumIsZero || newExpSInt <= 0) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(newExpSInt >= ((1 << expBits) - 1)) {
      c.exponent := ((1 << expBits) - 1)
      c.mantissa := 0
    } otherwise {
      c.exponent := newExpSInt.asUInt.resized
      c.mantissa := finalMantissa
    }
    
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
    when(isZero) {
      c.exponent := 0
      c.mantissa := 0
      c.sign := False
    } elsewhen(expSInt >= ((1 << expBits) - 1)) {
      c.exponent := ((1 << expBits) - 1)
      c.mantissa := 0
    } otherwise {
      c.exponent := expSInt.asUInt.resized
      c.mantissa := mantissa
    }
    
    c
  }
}
