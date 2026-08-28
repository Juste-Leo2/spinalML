package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._

/**
 * LinearLayer: A fully connected dense layer.
 * Formula: Y = Matmul(A, W) + b
 *
 * Weight-only quantization (wXaY): when `weightType` is an SInt format
 * (I4/I8), the weights are dequantized to the activation dtype through a
 * scaled cast before the float matmul:
 *   Y = Matmul(A, FloatML(W) * scales) + b
 * `weightScales` is either per-tensor (length 1) or per-channel
 * (length = number of weight stream beats, i.e. N columns).
 * Same-dtype weights keep the legacy direct wiring.
 */
case class LinearLayer[T <: Data, TW <: Data, TAcc <: Data](
  dataType: HardType[T],
  weightType: HardType[TW],
  accType: HardType[TAcc],
  shapeA: Seq[Int],
  shapeW: Seq[Int],
  lanes: Int,
  weightScales: Seq[Double] = Seq(1.0),
  tileSize: Int = 1024,
  parallelN: Boolean = false,
  temporal: Int = 0
) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  val N = shapeW(1)
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val w = slave(Tensor(weightType, shapeW, lanes))
    val b = slave(Tensor(accType, Seq(1, N), lanes = 1)) // Bias matches accumulator type
    val y = master(Tensor(accType, Seq(M, N), lanes = 1)) // Output can be processed sequentially
    // Command-boundary re-arm for the internal weight buffer (see MatmulOp)
    val reArm = in Bool()
  }
  
  // Weight-only quantization path: SInt weights feeding a FloatML activation
  // domain are dequantized before the (float) matmul. Any other combination
  // keeps the legacy direct wiring (uniform dtype matmul).
  private val needsDequant = (dataType(), weightType()) match {
    case (_: FloatML, _: SInt) => true
    case _ => false
  }
  
  val wForMatmul =
    if (needsDequant) cast(io.w, dataType, weightScales)
    else io.w.asInstanceOf[Tensor[T]]
  
  // 1. Matrix Multiplication: A * W_deq (reArm re-arms the internal B buffer,
  //    which carries this layer's weights). temporal > 0 bounds the rows in
  //    flight of the accumulator table (see MatmulOp).
  val matmulResult = matmul(io.a, wForMatmul, accType, parallelN = parallelN, reArm = Some(io.reArm), temporal = temporal)
  
  // 2. Add Bias (Broadcast): (A * W_deq) + b
  io.y <> bias_add(matmulResult, io.b)
}

object Linear {
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc], tileSize: Int, parallelN: Boolean, reArm: Option[Bool] = None, temporal: Int = 0): Tensor[TAcc] = {
    val comp = LinearLayer(a.dataType, a.dataType, accType, a.shape, w.shape, a.lanes, Seq(1.0), tileSize, parallelN, temporal)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.a <> a
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }

  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    apply(a, w, b, accType, 1024, false, None)
  }

  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T], tileSize: Int, parallelN: Boolean): Tensor[T] = {
    apply(a, w, b, a.dataType, tileSize, parallelN, None)
  }

  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(a, w, b, a.dataType, 1024, false, None)
  }

  // Weight-only quantization (wXaY): weights stored as SInt (I4/I8) plus
  // compile-time scale(s), activations in the float domain. No default
  // arguments here: only one overload of Linear may define defaults.
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[SInt], b: Tensor[TAcc], accType: HardType[TAcc], weightScales: Seq[Double], parallelN: Boolean, tileSize: Int, reArm: Option[Bool], temporal: Int): Tensor[TAcc] = {
    val comp = LinearLayer(a.dataType, w.dataType, accType, a.shape, w.shape, a.lanes, weightScales, tileSize, parallelN, temporal)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.a <> a
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
}
