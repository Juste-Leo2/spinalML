package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

object dot {
  /**
   * Dot product of two 1D vectors: a N-element vector dotted with a N-element vector.
   * Implemented as a thin wrapper on the validated matmul (M=1, N=1).
   */
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(a.shape.length == 1 && b.shape.length == 1, "Dot requires 1D tensors")
    require(a.shape == b.shape, s"Tensors must have the same shape. Original: ${a.shape}, Other: ${b.shape}")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    require(a.shape(0) % a.lanes == 0,
      s"Dot requires the vector length (${a.shape(0)}) to be a multiple of lanes (${a.lanes})")
    
    // View both vectors as 1xN and Nx1 matrices
    val aMatrix = reshape(a, Seq(1, a.shape(0)))
    val bMatrix = reshape(b, Seq(b.shape(0), 1))
    
    // matmul outputs a [1, 1] tensor with lanes = 1
    val scalar = matmul(aMatrix, bMatrix)
    
    // Back to a 1D scalar tensor
    reshape(scalar, Seq(1))
  }
}