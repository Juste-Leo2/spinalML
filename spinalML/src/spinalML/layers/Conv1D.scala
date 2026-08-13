package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * Conv1DLayer: A 1D Convolutional Layer (Single Input/Output Channel).
 * Formula: Y = Conv1D(X, W) + b
 */
case class Conv1DLayer[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], L_in: Int, K: Int, lanes: Int, tileSize: Int = 1024) extends Component {
  val L_out = L_in - K + 1
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L_in, 1), lanes = 1)) // Input Sequence
    val w = slave(Tensor(dataType, Seq(K, 1), lanes = K)) // Kernel Weights MUST match seq2col lanes
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(L_out, 1), lanes = 1)) // Output Sequence
  }
  
  // 1. Seq2Col: Convert input sequence into sliding windows
  // Output shape: [L_out, K], lanes = K
  val cols = seq2col(io.x, K)
  
  // 2. Matrix Multiplication: cols * W
  // cols is [L_out, K], W is [K, 1]. Output is [L_out, 1]
  val matmulResult = matmul(cols, io.w, accType, parallelN = false)
  
  // 3. Add Bias
  io.y <> bias_add(matmulResult, io.b)
}

object Conv1D {
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    val K = w.shape(0)
    val comp = Conv1DLayer(x.dataType, accType, x.shape(0), K, x.lanes, tileSize = K)
    comp.io.x <> x
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
  
  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(x, w, b, x.dataType)
  }
}
