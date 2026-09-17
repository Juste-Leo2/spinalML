// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.{RoundingConfig, RoundingMode}
import spinalML.tensors.Tensor

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
        // 1. Shift Arithmetic Right (+ RNE rounding unless Truncate legacy mode).
        // RNE on an arithmetic shift: truncated = floor(x / 2^shift),
        // guard = dropped bit shift-1, sticky = OR of dropped bits shift-2..0,
        // lsb = LSB of the kept result. roundUp on tie goes to even.
        val shifted: SInt = if (rounding == RoundingMode.Truncate || shift <= 0) {
          (valIn >> shift).resize(valIn.getWidth)
        } else {
          val inW = valIn.getWidth
          val truncated = (valIn >> shift).resize(inW)
          if (shift >= inW) {
            // All magnitude bits dropped: exact result is in (-1, 1).
            // Only valIn == 0 is exact; +/- tie rounds to even (0).
            val roundUp = (valIn =/= 0) && (valIn(inW - 1) === False || (valIn(inW - 2 downto 0) =/= 0))
            (truncated + Mux(roundUp, S(1, inW bits), S(0, inW bits))).resize(inW)
          } else {
            val guard = valIn(shift - 1)
            val sticky: Bool = if (shift >= 2) (valIn(shift - 2 downto 0) =/= 0) else False
            val lsb = truncated(0)
            val roundUp = guard && (sticky || lsb)
            (truncated + Mux(roundUp, S(1, inW bits), S(0, inW bits))).resize(inW)
          }
        }
        
        // 2. Saturation (Clamp) to target type limits
        val maxVal = (1 << (valOut.getWidth - 1)) - 1
        val minVal = -(1 << (valOut.getWidth - 1))
        
        val saturated = Mux(shifted > maxVal, S(maxVal, valIn.getWidth bits),
                          Mux(shifted < minVal, S(minVal, valIn.getWidth bits),
                              shifted))
                              
        io.c.stream.payload(i).assignFrom(saturated.resize(valOut.getWidth))
        
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
