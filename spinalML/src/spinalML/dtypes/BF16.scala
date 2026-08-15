package spinalML.dtypes

import spinal.core._

object BF16 {
  /**
   * Bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits.
   * Total 16 bits. Excellent dynamic range for ML.
   */
  def apply(): FloatML = FloatML(expBits = 8, mantBits = 7)
}
