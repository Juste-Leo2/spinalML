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
}
