package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

object reshape {
  /**
   * Reshapes a tensor to a new shape.
   * This is a metadata operation that costs 0 logic cells.
   * Note: The new shape must have the same total number of elements.
   */
  def apply[T <: Data](a: Tensor[T], newShape: Seq[Int]): Tensor[T] = {
    require(a.shape.product == newShape.product, s"Reshape cannot change total number of elements. Original: ${a.shape}, New: $newShape")
    require(newShape.product % a.lanes == 0, s"New shape elements (${newShape.product}) must be a multiple of lanes (${a.lanes})")
    
    val c = Tensor(a.dataType, newShape, a.lanes)
    c.stream << a.stream
    c
  }
}
