// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

/**
 * RequantizeOp: Shifts and saturates values from a larger accumulator type (e.g. I32)
 * to a smaller target type (e.g. I8 or I16).
 */
case class RequantizeOp[TIn <: Data, TOut <: Data](
  dataTypeIn: HardType[TIn],
  dataTypeOut: HardType[TOut],
  shape: Seq[Int],
  lanes: Int,
  shift: Int
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
        // 1. Shift Arithmetic Right
        val shifted = (valIn >> shift).resize(valIn.getWidth)
        
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
  def apply[TIn <: Data, TOut <: Data](a: Tensor[TIn], dataTypeOut: HardType[TOut], shift: Int): Tensor[TOut] = {
    val comp = RequantizeOp(a.dataType, dataTypeOut, a.shape, a.lanes, shift)
    comp.io.a <> a
    comp.io.c
  }
}
