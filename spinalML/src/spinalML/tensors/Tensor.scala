package spinalML.tensors

import spinal.core._

/**
 * Generic Tensor representation in hardware.
 * Uses a flat Vec to store elements and mathematical indexing for multi-dimensional access.
 */
case class Tensor[T <: Data](dataType: HardType[T], shape: Seq[Int]) extends Bundle {
  require(shape.nonEmpty, "Tensor shape cannot be empty")
  require(shape.forall(_ > 0), "All dimensions must be > 0")

  val totalElements = shape.product
  val data = Vec(dataType, totalElements)

  /**
   * Helper to compute the flat index from multi-dimensional indices
   * (Row-major order format)
   */
  def getFlatIndex(indices: Seq[Int]): Int = {
    require(indices.length == shape.length, s"Expected ${shape.length} indices, got ${indices.length}")
    var flatIndex = 0
    var stride = 1
    for (i <- shape.indices.reverse) {
      val index = indices(i)
      require(index >= 0 && index < shape(i), s"Index $index out of bounds for dimension $i (size ${shape(i)})")
      flatIndex += index * stride
      stride *= shape(i)
    }
    flatIndex
  }

  /**
   * Access an element at specific multi-dimensional indices
   */
  def apply(indices: Int*): T = {
    data(getFlatIndex(indices))
  }
}

object Tensor {
  // GGML inspired constructors
  
  def Tensor1D[T <: Data](dataType: HardType[T], ne0: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0))
  }

  def Tensor2D[T <: Data](dataType: HardType[T], ne0: Int, ne1: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0, ne1))
  }

  def Tensor3D[T <: Data](dataType: HardType[T], ne0: Int, ne1: Int, ne2: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0, ne1, ne2))
  }
}
