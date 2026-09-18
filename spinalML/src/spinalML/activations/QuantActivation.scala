// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import spinal.core._
import spinalML.utils.{MathLUTs, UnaryLUTOp}
import spinalML.RoundingMode

/**
 * Quantized activation LUT builder following the TFLite int8/uint8 conventions
 * (the only reference ONNX core does not define):
 *
 *   x = (q - inputZeroPoint) * inputScale
 *   out_code = saturate_round(f(x) / outputScale + outputZeroPoint)
 *
 * The ROM tabulates every input code (2^bitWidth entries), so the output is
 * exact for the defined real function under the elaboration rounding switch
 * (RNE default, trunc = legacy half-up). TFLite kernels use half-away rounding
 * in their fixed-point multipliers; this divergence is the same documented
 * one as `QuantizeLinear` (see docs/rounding_policy.md §5).
 */
object QuantActivation {

  /** TFLite logistic output quantization (LOGISTIC). */
  val logisticOutputScale: Double = 1.0 / 256.0
  def logisticOutputZeroPoint(signed: Boolean): Int = if (signed) -128 else 0

  /** TFLite tanh output quantization (TANH). */
  val tanhOutputScale: Double = 1.0 / 128.0
  def tanhOutputZeroPoint(signed: Boolean): Int = if (signed) 0 else 128

  def lut[T <: Data](
    dataType: HardType[T],
    shape: Seq[Int],
    lanes: Int,
    inputScale: Double,
    inputZeroPoint: Int,
    outputScale: Double,
    outputZeroPoint: Int,
    rounding: RoundingMode
  )(f: Double => Double): UnaryLUTOp[T] = {
    require(inputScale > 0.0, s"QuantActivation: inputScale must be > 0, got $inputScale")
    require(outputScale > 0.0, s"QuantActivation: outputScale must be > 0, got $outputScale")
    val bitWidth = dataType.getBitsWidth
    val signed = dataType().isInstanceOf[SInt]
    val valFn = if (signed) MathLUTs.intValFn(bitWidth) else MathLUTs.uintValFn(bitWidth)
    val encodeFn = if (signed) MathLUTs.intEncodeFn(bitWidth, rounding)
                   else MathLUTs.uintEncodeFn(bitWidth, rounding)
    UnaryLUTOp(dataType, shape, lanes, valFn, encodeFn,
      q => f((q - inputZeroPoint) * inputScale) / outputScale + outputZeroPoint)
  }
}
