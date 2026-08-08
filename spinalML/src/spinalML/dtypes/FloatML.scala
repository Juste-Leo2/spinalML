package spinalML.dtypes

import spinal.core._

/**
 * Generic Floating Point representation for Hardware Machine Learning.
 * Omits NaNs and subnormals logic to optimize FPGA synthesis.
 */
case class FloatML(expBits: Int, mantBits: Int) extends Bundle {
  val sign = Bool()
  val exponent = UInt(expBits bits)
  val mantissa = UInt(mantBits bits)
  
  // IEEE bias calculation: (2^(expBits - 1)) - 1
  def bias: Int = (1 << (expBits - 1)) - 1
}
