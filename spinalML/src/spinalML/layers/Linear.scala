package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * LinearLayer: A fully connected dense layer.
 * Formula: Y = Matmul(A, W) + b
 */
case class LinearLayer[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], shapeA: Seq[Int], shapeW: Seq[Int], lanes: Int, tileSize: Int = 1024, parallelN: Boolean = false) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  val N = shapeW(1)
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val w = slave(Tensor(dataType, shapeW, lanes))
    val b = slave(Tensor(accType, Seq(1, N), lanes = 1)) // Bias matches accumulator type
    val y = master(Tensor(accType, Seq(M, N), lanes = 1)) // Output can be processed sequentially
  }
  
  // 1. Matrix Multiplication: A * W
  val matmulResult = matmul(io.a, io.w, accType, parallelN = parallelN)
  
  // 2. Add Bias (Broadcast): (A * W) + b
  io.y <> bias_add(matmulResult, io.b)
}

object Linear {
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc], tileSize: Int, parallelN: Boolean): Tensor[TAcc] = {
    val comp = LinearLayer(a.dataType, accType, a.shape, w.shape, a.lanes, tileSize, parallelN)
    comp.io.a <> a
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
  
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    apply(a, w, b, accType, 1024, false)
  }
  
  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T], tileSize: Int, parallelN: Boolean): Tensor[T] = {
    apply(a, w, b, a.dataType, tileSize, parallelN)
  }
  
  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(a, w, b, a.dataType, 1024, false)
  }
}
