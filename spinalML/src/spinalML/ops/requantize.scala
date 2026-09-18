// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.{RoundingConfig, RoundingMode}
import spinalML.tensors.Tensor

/**
 * Shared integer-narrowing datapath: arithmetic shift right with the
 * elaboration rounding policy, then two's-complement saturation. Single
 * implementation reused by [[RequantizeOp]] and the AvgPool* integer paths so
 * every integer narrowing in the engine rounds identically (test goldens
 * parameterize one semantics, not a per-op copy).
 */
object RequantizeMath {

  /**
   * SInt narrowing: RNE (guard + sticky, tie-to-even) or legacy truncation,
   * then clamp to the `outWidth` two's-complement range.
   *
   * RNE on an arithmetic shift: truncated = floor(x / 2^shift), guard = dropped
   * bit shift-1, sticky = OR of dropped bits shift-2..0, lsb = LSB of the kept
   * result. Round-up on tie goes to even.
   */
  def shiftSaturate(value: SInt, shift: Int, outWidth: Int, rounding: RoundingMode): SInt = {
    val inW = value.getWidth
    val shifted: SInt = if (rounding == RoundingMode.Truncate || shift <= 0) {
      (value >> shift).resize(inW)
    } else {
      val truncated = (value >> shift).resize(inW)
      if (shift >= inW) {
        // All magnitude bits dropped: exact result is in (-1, 1).
        // Only value == 0 is exact; +/- tie rounds to even (0).
        val roundUp = (value =/= 0) && (value(inW - 1) === False || (value(inW - 2 downto 0) =/= 0))
        (truncated + Mux(roundUp, S(1, inW bits), S(0, inW bits))).resize(inW)
      } else {
        val guard = value(shift - 1)
        val sticky: Bool = if (shift >= 2) (value(shift - 2 downto 0) =/= 0) else False
        val lsb = truncated(0)
        val roundUp = guard && (sticky || lsb)
        (truncated + Mux(roundUp, S(1, inW bits), S(0, inW bits))).resize(inW)
      }
    }

    val maxVal = (1 << (outWidth - 1)) - 1
    val minVal = -(1 << (outWidth - 1))

    val saturated = Mux(shifted > maxVal, S(maxVal, inW bits),
      Mux(shifted < minVal, S(minVal, inW bits),
        shifted))
    saturated.resize(outWidth)
  }

  /**
   * UInt narrowing: same RNE/truncation policy, no saturation. The only
   * current user is AvgPool, whose window average is mathematically bounded by
   * the input range, so a clamp would be dead logic. `shift < value.getWidth`
   * always holds there (accumulator width is `w + shift`).
   */
  def shiftRound(value: UInt, shift: Int, outWidth: Int, rounding: RoundingMode): UInt = {
    require(shift >= 0 && shift < value.getWidth,
      s"RequantizeMath.shiftRound expects 0 <= shift < input width, got shift=$shift width=${value.getWidth}")
    val inW = value.getWidth
    val shifted: UInt = if (rounding == RoundingMode.Truncate || shift == 0) {
      (value >> shift).resize(inW)
    } else {
      val truncated = (value >> shift).resize(inW)
      val guard = value(shift - 1)
      val sticky: Bool = if (shift >= 2) (value(shift - 2 downto 0) =/= 0) else False
      val roundUp = guard && (sticky || truncated(0))
      (truncated + Mux(roundUp, U(1, inW bits), U(0, inW bits))).resize(inW)
    }
    shifted.resize(outWidth)
  }
}

/**
 * RequantizeOp: Shifts and saturates values from a larger accumulator type (e.g. I32)
 * to a smaller target type (e.g. I8 or I16).
 *
 * @param rounding elaboration-only rounding policy (RNE by default, Truncate =
 *                 legacy bit-exact). In Truncate mode the RNE datapath is not
 *                 elaborated (0 LUT added).
 */
case class RequantizeOp[TIn <: Data, TOut <: Data](
  dataTypeIn: HardType[TIn],
  dataTypeOut: HardType[TOut],
  shape: Seq[Int],
  lanes: Int,
  shift: Int,
  rounding: RoundingMode = RoundingConfig.current
) extends Component {

  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, shape, lanes))
    val c = master(Tensor(dataTypeOut, shape, lanes))
  }

  // Pass through the stream control signals
  io.c.stream.arbitrationFrom(io.a.stream)

  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.c.stream.payload(i)) match {
      case (valIn: SInt, valOut: SInt) =>
        io.c.stream.payload(i).assignFrom(
          RequantizeMath.shiftSaturate(valIn, shift, valOut.getWidth, rounding))

      case _ =>
        throw new Exception("RequantizeOp currently only supports SInt -> SInt conversions.")
    }
  }
}

object requantize {
  def apply[TIn <: Data, TOut <: Data](
    a: Tensor[TIn],
    dataTypeOut: HardType[TOut],
    shift: Int
  ): Tensor[TOut] =
    apply(a, dataTypeOut, shift, RoundingConfig.current)

  def apply[TIn <: Data, TOut <: Data](
    a: Tensor[TIn],
    dataTypeOut: HardType[TOut],
    shift: Int,
    rounding: RoundingMode
  ): Tensor[TOut] = {
    val comp = RequantizeOp(a.dataType, dataTypeOut, a.shape, a.lanes, shift, rounding)
    comp.io.a <> a
    comp.io.c
  }
}
