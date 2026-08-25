package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * Conv1DLayer: A 1D Convolutional Layer (Single Input/Output Channel).
 * Formula: Y = Conv1D(X, W) + b
 */
case class Conv1DLayer[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], L_in: Int, inChannels: Int, outChannels: Int, K: Int, outLanes: Int, tileSize: Int = 1024, parallelN: Boolean = false) extends Component {
  val L_out = L_in - K + 1

  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L_in, inChannels), lanes = 1)) // Input Sequence
    val w = slave(Tensor(dataType, Seq(K * inChannels, outChannels), lanes = outLanes)) // Kernel Weights
    val b = slave(Tensor(accType, Seq(1, outChannels), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(L_out, outChannels), lanes = 1)) // Output Sequence
    // Command-boundary re-arm for the internal weight buffer (see MatmulOp)
    val reArm = in Bool()
  }

  // 1. Seq2Col: Convert input sequence into sliding windows
  // Output shape: [L_out, K * inChannels], lanes = outLanes
  val cols = seq2col(io.x, K, outLanes)

  // 2. Matrix Multiplication: cols * W (reArm re-arms the internal B buffer,
  //    which carries this layer's weights)
  // cols is [L_out, K * inChannels], W is [K * inChannels, outChannels]. Output is [L_out, outChannels]
  val matmulResult = matmul(cols, io.w, accType, parallelN = parallelN, reArm = Some(io.reArm))

  // 3. Add Bias
  io.y <> bias_add(matmulResult, io.b)
}

object Conv1D {
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc], parallelN: Boolean = false, reArm: Option[Bool] = None): Tensor[TAcc] = {
    val inChannels = if (x.shape.length == 2) x.shape(1) else 1
    val outChannels = w.shape(1)
    val K = w.shape(0) / inChannels

    val comp = Conv1DLayer(x.dataType, accType, x.shape(0), inChannels, outChannels, K, outLanes = w.lanes, tileSize = w.shape(0), parallelN = parallelN)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.x <> x
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
  
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    apply(x, w, b, accType, false, None)
  }

  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T], parallelN: Boolean): Tensor[T] = {
    apply(x, w, b, x.dataType, parallelN, None)
  }

  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(x, w, b, x.dataType, false, None)
  }
}
