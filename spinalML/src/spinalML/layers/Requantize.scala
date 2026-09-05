// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

case class RequantizeLayer[T <: Data, TOut <: Data](
  dataTypeIn: HardType[T],
  dataTypeOut: HardType[TOut],
  shape: Seq[Int],
  lanes: Int,
  shift: Int
) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataTypeIn, shape, lanes))
    val y = master(Tensor(dataTypeOut, shape, lanes))
  }
  
  io.y <> requantize(io.x, dataTypeOut, shift)
}

object RequantizeLayer {
  def apply[T <: Data, TOut <: Data](x: Tensor[T], dataTypeOut: HardType[TOut], shift: Int): Tensor[TOut] = {
    val comp = new RequantizeLayer(x.dataType, dataTypeOut, x.shape, x.lanes, shift)
    comp.io.x <> x
    comp.io.y
  }
}
