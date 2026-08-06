package spinalML.examples

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import spinalML.ops._

/**
 * SimplePipeline: An example of a complete AI operation chain.
 * Formula: Y = Matmul(A + B, W)
 * 
 * Dimensions (optimized for fast testing):
 * - A and B: [1, 2] (Vectors of size 2)
 * - W: [2, 1] (Vector of size 2)
 * - Y: [1, 1] (Scalar output)
 */
case class SimplePipeline[T <: Data](dataType: HardType[T], lanes: Int = 2) extends Component {
  val M = 1
  val K = 2
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(M, K), lanes))
    val b = slave(Tensor(dataType, Seq(M, K), lanes))
    val w = slave(Tensor(dataType, Seq(K, 1), lanes))
    val y = master(Tensor(dataType, Seq(M, 1), lanes = 1))
  }
  
  // 1. Element-wise Addition: sum = A + B
  val sum = add(io.a, io.b)
  
  // 2. Matrix Multiplication (Tiled Double-Buffered): Y = sum * W
  // We use tileSize = 2 so the whole K dimension fits in one tile.
  io.y <> matmul(sum, io.w, tileSize = 2)
}
