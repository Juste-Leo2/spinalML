package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * Conv2DLayer: A 2D Convolutional Layer (Single Input/Output Channel).
 * Formula: Y = Conv2D(X, W) + b
 */
case class Conv2DLayer[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], H: Int, W_in: Int, K: Int, tileSize: Int = 1024) extends Component {
  val H_out = H - K + 1
  val W_out = W_in - K + 1
  val totalWindows = H_out * W_out
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(H, W_in), lanes = 1)) // Input Image
    val w = slave(Tensor(dataType, Seq(K * K, 1), lanes = K * K)) // Flattened Kernel Weights
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(H_out, W_out), lanes = 1)) // 2D Output Image
  }
  
  // 1. Im2Col: Convert 2D image into sliding windows
  // Output shape: [totalWindows, K*K], lanes = K*K
  val cols = im2col(io.x, K)
  
  // 2. Matrix Multiplication: cols * W
  // cols is [L_out, K*K], W is [K*K, 1]. Output is [L_out, 1]
  val matmulResult = matmul(cols, io.w, accType, parallelN = false)
  
  // 3. Add Bias
  val biasAdded = bias_add(matmulResult, io.b)
  
  // 4. Reshape to 2D
  io.y <> reshape(biasAdded, Seq(H_out, W_out))
}

object Conv2D {
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    // Assuming w.shape(0) == K * K
    // To extract K, we take the square root. But since K is a hardware parameter,
    // we can deduce it from the lanes or shape.
    val K2 = w.shape(0)
    val K = Math.sqrt(K2).toInt
    require(K * K == K2, "Kernel size K*K must be a perfect square")
    
    val comp = Conv2DLayer(x.dataType, accType, x.shape(0), x.shape(1), K, tileSize = K2)
    comp.io.x <> x
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
  
  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(x, w, b, x.dataType)
  }
}
