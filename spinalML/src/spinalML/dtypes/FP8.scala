package spinalML.dtypes

object FP8_E4M3 {
  /**
   * FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
   * Total 8 bits. Good for ML bandwidth.
   */
  def apply(): FloatML = FloatML(expBits = 4, mantBits = 3)
}

object FP8_E5M2 {
  /**
   * FP8 E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits.
   */
  def apply(): FloatML = FloatML(expBits = 5, mantBits = 2)
}
