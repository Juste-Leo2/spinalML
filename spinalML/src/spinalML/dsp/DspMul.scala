// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dsp

import spinal.core._
import spinalML.arithmetic.HardwareMul

/**
 * Hardware DSP Multiplier Facade.
 *
 * Preserves 100% backward compatibility with all existing pipeline and tensor operations
 * while routing physical synthesis to the unified [[HardwareMul]] engine.
 */
object DspMul {

  /** Multiplies two hardware signals with explicit accumulator return type. */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    enable: Bool,
    accType: HardType[TAcc],
    latency: Int,
    dspConfig: DspConfig
  ): TAcc = HardwareMul(a, b, enable, accType, latency, dspConfig)

  /** Overload inferring output accumulator type from input operand type. */
  def apply[T <: Data](
    a: T,
    b: T,
    enable: Bool,
    latency: Int,
    dspConfig: DspConfig
  ): T = HardwareMul(a, b, enable, latency, dspConfig)

  /** Convenience overload with default enable = True and default DspConfig. */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    accType: HardType[TAcc]
  ): TAcc = HardwareMul(a, b, accType)
}
