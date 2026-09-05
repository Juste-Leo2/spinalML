// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

object flatten {
  /**
   * Flattens a multi-dimensional tensor into a 1D tensor.
   * This is a metadata operation that costs 0 logic cells.
   */
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val totalElements = a.shape.product
    val c = Tensor(a.dataType, Seq(totalElements), a.lanes)
    c.stream << a.stream
    c
  }
}
