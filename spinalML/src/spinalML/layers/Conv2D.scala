package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * Conv2DLayer: A 2D Convolutional Layer (Single Input/Output Channel).
 * Formula: Y = Conv2D(X, W) + b
 */
case class Conv2DLayer[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], H: Int, W_in: Int, inChannels: Int, outChannels: Int, K: Int, outLanes: Int, tileSize: Int = 1024, parallelN: Boolean = false, temporal: Int = 0) extends Component {
  val H_out = H - K + 1
  val W_out = W_in - K + 1
  val totalWindows = H_out * W_out

  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(H, W_in, inChannels), lanes = 1)) // Input Image [H, W, C]
    val w = slave(Tensor(dataType, Seq(K * K * inChannels, outChannels), lanes = outLanes)) // Kernel Weights
    val b = slave(Tensor(accType, Seq(1, outChannels), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(H_out, W_out, outChannels), lanes = 1)) // Output Image
    // Command-boundary re-arm for the internal weight buffer (see MatmulOp)
    val reArm = in Bool()
  }

  // 1. Im2Col: Convert 2D image into sliding windows
  // Output shape: [totalWindows, K*K*inChannels], lanes = outLanes
  val cols = im2col(io.x, K, outLanes)

  // 2. Matrix Multiplication: cols * W (reArm re-arms the internal B buffer,
  //    which carries this layer's weights). temporal > 0 bounds the rows in
  //    flight of the accumulator table (see MatmulOp).
  val matmulResult = matmul(cols, io.w, accType, parallelN = parallelN, reArm = Some(io.reArm), temporal = temporal)

  // 3. Add Bias
  val biasAdded = bias_add(matmulResult, io.b)

  // 4. Reshape to 3D [H_out, W_out, outChannels]
  io.y <> reshape(biasAdded, Seq(H_out, W_out, outChannels))
}

object Conv2D {
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc], parallelN: Boolean = false, reArm: Option[Bool] = None, temporal: Int = 0): Tensor[TAcc] = {
    val inChannels = if (x.shape.length == 3) x.shape(2) else 1
    val outChannels = w.shape(1)

    val K2C = w.shape(0)
    val K2 = K2C / inChannels
    val K = Math.sqrt(K2).toInt
    require(K * K * inChannels == K2C, "Kernel weights shape must be K*K*inChannels")

    val comp = Conv2DLayer(x.dataType, accType, x.shape(0), x.shape(1), inChannels, outChannels, K, outLanes = w.lanes, tileSize = K2C, parallelN = parallelN, temporal = temporal)
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
