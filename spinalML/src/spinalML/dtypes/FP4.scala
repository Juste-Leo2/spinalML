package spinalML.dtypes

import spinal.core._

object FP4_E2M1 {
  def apply(): FloatML = FloatML(expBits = 2, mantBits = 1)
}
