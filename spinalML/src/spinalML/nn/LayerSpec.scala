// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._

/**
 * LayerSpec describes a Neural Network layer in a declarative way.
 * It is completely independent of the Sequential topology, allowing future reuse in 
 * complex graphs like ResNets or Transformers (Attention blocks).
 */
trait LayerSpec {
  def outType(default: HardType[Data]): HardType[Data] = default
  def weightType(default: HardType[Data]): HardType[Data] = default
  
  // Predicts the output shape given an input shape
  def getOutShape(inShape: Seq[Int]): Seq[Int]
  
  // Computes the number of elements required for weights and biases
  def getWeightShape(): Seq[Int]
  def getBiasShape(): Seq[Int]
}

/**
 * K-pass spill contract shared by GEMM-like layers (Linear, Conv2D, ...).
 * The flattened K axis (`spillKFull`: inFeatures, or K*K*inChannels) streams
 * one `spillKSlice` slice per pass (P = KFull/Ks passes, M*N full-width
 * partials in DDR between passes). -1 (default) = no spill, legacy one-shot.
 * Concrete layers add `require`s: Ks divides KFull and is a multiple of
 * effLanes (pass-internal chunking must match the replica fadd order).
 */
trait SpillableGEMM extends LayerSpec {
  def spillKSlice: Int
  def spillKFull: Int
  def spillN: Int
  def spilling: Boolean = spillKSlice > 0
  def spillPasses: Int = if (spillKSlice <= 0) 1 else spillKFull / spillKSlice
}

/**
 * 2D convolution layer. Weight/bias dtypes default to the pipeline dtype;
 * `customWeightType` enables narrow integer weights (e.g. true I4 nibbles),
 * which Sequential sign-extends to the activation width so the integer
 * matmul consumes them. Mixed precision on Conv2D is therefore supported
 * only in the integer domain — unlike Linear, there is no float-dequant
 * path (float activations require float weights). See
 * docs/bugs/2026-08-w4a8-session.md.
 */
case class Conv2D(
  inChannels: Int, 
  outChannels: Int, 
  kernelSize: Int,
  customType: Option[HardType[Data]] = None,
  customWeightType: Option[HardType[Data]] = None,
  // K-axis chunk width of the matmul weight beats (M2 pattern, mirrors
  // Linear.weightLanes): the weight memory layout is unchanged (one
  // flattened K*K*inChannels row per output channel); only the per-beat lane
  // count and the matmul's internal K chunking change.
  // -1 (default) = kernelSize*kernelSize, the legacy full row-group width.
  // Any other value must divide K*K*inChannels (dense beats == column
  // groups, no zero-padding — see the OPS-07 note in Sequential).
  weightLanes: Int = -1,
  lanes: Int = 1,
  // P1 compute-side spill (docs/ddr_spill_ops.md): K-slice width streamed per
  // pass over the flattened K*K*inChannels axis (same gabarit as Linear).
  // -1 (default) = no spill, legacy one-shot convolution.
  spillKSlice: Int = -1
) extends SpillableGEMM {
  require(weightLanes == -1 || (weightLanes > 0 && (kernelSize * kernelSize * inChannels) % weightLanes == 0),
    s"Conv2D weightLanes=$weightLanes must be -1 or a positive divisor of K*K*inChannels=${kernelSize * kernelSize * inChannels}")
  require(spillKSlice == -1 || (spillKSlice > 0 && (kernelSize * kernelSize * inChannels) % spillKSlice == 0),
    s"Conv2D spillKSlice=$spillKSlice must be -1 or a positive divisor of K*K*inChannels=${kernelSize * kernelSize * inChannels}")
  require(spillKSlice == -1 || spillKSlice % effLanes == 0,
    s"Conv2D spillKSlice=$spillKSlice must be a multiple of effLanes=$effLanes " +
      "(pass-internal chunking must match the replica fadd order exactly)")
  def spillKFull: Int = kernelSize * kernelSize * inChannels
  def spillN: Int = outChannels

  /** Effective per-beat width: K*K when the default (-1) is left untouched. */
  def effLanes: Int = if (weightLanes <= 0) kernelSize * kernelSize else weightLanes
  override def outType(default: HardType[Data]) = customType.getOrElse(default)
  override def weightType(default: HardType[Data]) = customWeightType.getOrElse(default)
  
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2, "Conv2D requires at least 2D input shape (H, W)")
    val h = inShape(0)
    val w = inShape(1)
    val hOut = h - kernelSize + 1
    val wOut = w - kernelSize + 1
    // Preserving spatial topology: (H_out, W_out, C_out)
    Seq(hOut, wOut, outChannels)
  }
  
  override def getWeightShape(): Seq[Int] = Seq(kernelSize * kernelSize * inChannels, outChannels)
  override def getBiasShape(): Seq[Int] = Seq(1, outChannels)
}

case class ReLU() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0) // No weights
  override def getBiasShape(): Seq[Int] = Seq(0)   // No bias
}

case class Linear(
  inFeatures: Int, 
  outFeatures: Int,
  customType: Option[HardType[Data]] = None,
  customWeightType: Option[HardType[Data]] = None,
  weightScales: Seq[Double] = Seq(1.0),
  // K-axis chunk width of the matmul weight/activation beats (M2): the
  // weight memory layout is unchanged (linear [inFeatures x outFeatures]);
  // only the per-beat lane count and the matmul's internal K chunking
  // change. -1 (default) = inFeatures, the legacy full-width beats.
  weightLanes: Int = -1,
  lanes: Int = 1,
  // S0 compute-side spill (docs/ddr_final_impl.md): K-slice width streamed per
  // pass (P = inFeatures / spillKSlice passes, M*N full-width partials in DDR
  // between passes). -1 (default) = no spill, legacy one-shot GEMM.
  spillKSlice: Int = -1
) extends SpillableGEMM {
  require(weightLanes == -1 || (weightLanes > 0 && inFeatures % weightLanes == 0),
    s"Linear weightLanes=$weightLanes must be -1 or a positive divisor of inFeatures=$inFeatures")
  require(spillKSlice == -1 || (spillKSlice > 0 && inFeatures % spillKSlice == 0),
    s"Linear spillKSlice=$spillKSlice must be -1 or a positive divisor of inFeatures=$inFeatures")
  require(spillKSlice == -1 || spillKSlice % effLanes == 0,
    s"Linear spillKSlice=$spillKSlice must be a multiple of effLanes=$effLanes " +
      "(pass-internal chunking must match the replica fadd order exactly)")

  /** Effective per-beat width: inFeatures when the default (-1) is left untouched. */
  def effLanes: Int = if (weightLanes <= 0) inFeatures else weightLanes
  def spillKFull: Int = inFeatures
  def spillN: Int = outFeatures
  override def outType(default: HardType[Data]) = customType.getOrElse(default)
  override def weightType(default: HardType[Data]) = customWeightType.getOrElse(default)
  
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.last == inFeatures,
      s"Linear expects its input's last dimension to be inFeatures ($inFeatures), got shape $inShape")
    inShape.dropRight(1) :+ outFeatures
  }
  override def getWeightShape(): Seq[Int] = Seq(inFeatures, outFeatures)
  override def getBiasShape(): Seq[Int] = Seq(1, outFeatures)
}

case class Conv1D(
  inChannels: Int,
  outChannels: Int,
  kernelSize: Int,
  customType: Option[HardType[Data]] = None,
  customWeightType: Option[HardType[Data]] = None,
  // Same M2 pattern as Conv2D: -1 (default) = kernelSize*inChannels, the
  // legacy width; otherwise must divide K*inChannels (no padding).
  weightLanes: Int = -1,
  lanes: Int = 1,
  // P2 compute-side spill (docs/ddr_spill_ops.md): K-slice width streamed per
  // pass over the flattened K*inChannels axis (same gabarit as Conv2D P1).
  // -1 (default) = no spill, legacy one-shot convolution.
  spillKSlice: Int = -1
) extends SpillableGEMM {
  require(weightLanes == -1 || (weightLanes > 0 && (kernelSize * inChannels) % weightLanes == 0),
    s"Conv1D weightLanes=$weightLanes must be -1 or a positive divisor of K*inChannels=${kernelSize * inChannels}")
  require(spillKSlice == -1 || (spillKSlice > 0 && (kernelSize * inChannels) % spillKSlice == 0),
    s"Conv1D spillKSlice=$spillKSlice must be -1 or a positive divisor of K*inChannels=${kernelSize * inChannels}")
  require(spillKSlice == -1 || spillKSlice % effLanes == 0,
    s"Conv1D spillKSlice=$spillKSlice must be a multiple of effLanes=$effLanes " +
      "(pass-internal chunking must match the replica fadd order exactly)")
  def spillKFull: Int = kernelSize * inChannels
  def spillN: Int = outChannels

  /** Effective per-beat width: K*inChannels when the default (-1) is left untouched. */
  def effLanes: Int = if (weightLanes <= 0) kernelSize * inChannels else weightLanes
  override def outType(default: HardType[Data]) = customType.getOrElse(default)
  override def weightType(default: HardType[Data]) = customWeightType.getOrElse(default)
  
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2, "Conv1D requires at least 2D input shape (L, C)")
    val l = inShape(0)
    val lOut = l - kernelSize + 1
    Seq(lOut, outChannels)
  }
  
  override def getWeightShape(): Seq[Int] = Seq(kernelSize * inChannels, outChannels)
  override def getBiasShape(): Seq[Int] = Seq(1, outChannels)
}

case class LeakyReLU(shift: Int = 2) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Softmax(lanes: Int = 1) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class BatchNorm1D(
  features: Int,
  lanes: Int = -1,
  shift: Int = 0,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(features) // gamma
  override def getBiasShape(): Seq[Int] = Seq(features) // beta
}

case class LayerNorm1D(features: Int, lanes: Int = -1) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(features) // gamma
  override def getBiasShape(): Seq[Int] = Seq(features) // beta
}

case class MaxPool1D(poolSize: Int, stride: Int, lanes: Int = -1) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2, "MaxPool1D requires at least 2D input shape (L, C)")
    val l = inShape(0)
    val c = inShape(1)
    val lOut = (l - poolSize) / stride + 1
    Seq(lOut, c)
  }
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class AvgPool1D(
  poolSize: Int,
  stride: Int,
  lanes: Int = -1,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2, "AvgPool1D requires at least 2D input shape (L, C)")
    val l = inShape(0)
    val c = inShape(1)
    val lOut = (l - poolSize) / stride + 1
    Seq(lOut, c)
  }
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class MaxPool2D(poolSize: Int, stride: Int, lanes: Int = 1) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2 && inShape.length <= 3, "MaxPool2D requires a 2D (H, W) or 3D (H, W, C) input shape")
    val h = inShape(0)
    val w = inShape(1)
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    if (inShape.length == 3) Seq(hOut, wOut, inShape(2)) else Seq(hOut, wOut)
  }
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class AvgPool2D(
  poolSize: Int,
  stride: Int,
  lanes: Int = 1,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  require(isPow2(poolSize * poolSize), "AvgPool2D requires isPow2(poolSize*poolSize) (shift-based division)")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2 && inShape.length <= 3, "AvgPool2D requires a 2D (H, W) or 3D (H, W, C) input shape")
    val h = inShape(0)
    val w = inShape(1)
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    if (inShape.length == 3) Seq(hOut, wOut, inShape(2)) else Seq(hOut, wOut)
  }
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

/**
 * Quantized activation specs. On 8-bit integers, `inputScale`/`inputZeroPoint`
 * describe the input tensor quantization (`x = (q - zp) * scale`); the output
 * quantization follows the TFLite conventions (LOGISTIC scale 1/256, zp -128
 * int8 / 0 uint8; TANH scale 1/128, zp 0 int8 / 128 uint8). FloatML ignores
 * both (pure real sigmoid/tanh).
 */
case class Sigmoid(
  inputScale: Double = 1.0,
  inputZeroPoint: Int = 0,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  require(inputScale > 0.0, s"Sigmoid inputScale must be > 0, got $inputScale")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Tanh(
  inputScale: Double = 1.0,
  inputZeroPoint: Int = 0,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  require(inputScale > 0.0, s"Tanh inputScale must be > 0, got $inputScale")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

/**
 * Cast to `targetType`. When casting an SInt tensor to a FloatML type,
 * `scales` implements the dequantization step of integer-domain pipelines:
 * W_float = FloatML(W_int) * scale (per-tensor, length 1). Default Seq(1.0)
 * keeps the pure cast behavior.
 */
case class Cast(
  targetType: HardType[Data],
  scales: Seq[Double] = Seq(1.0),
  runtimeScale: Boolean = false,
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
  override def outType(default: HardType[Data]) = targetType
}

object Cast {
  def runtime(targetType: HardType[Data]): Cast = Cast(targetType, runtimeScale = true)
}

case class Flatten() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = Seq(1, inShape.product)
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Requantize(
  shift: Int,
  targetType: HardType[Data],
  rounding: Option[spinalML.RoundingMode] = None
) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
  override def outType(default: HardType[Data]) = targetType
}

case class Repack(newLanes: Int) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

/**
 * DAG merge nodes: consume two earlier tensors by node index (position in the
 * modelSpec, where node 0 is the network input and node k is the output of the
 * k-th spec entry). References must point strictly backwards, which makes the
 * graph acyclic by construction.
 *
 * Their real shape/type inference is performed by the Sequential builder, which
 * knows the shapes of both referenced nodes; getOutShape is therefore unused.
 */
case class Add(a: Int, b: Int) extends LayerSpec {
  require(a >= 0 && b >= 0, "Add node references must be non-negative")
  require(a != b, "Add requires two distinct nodes")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Concat(a: Int, b: Int, axis: Int = 0, lanes: Int = 1) extends LayerSpec {
  require(a >= 0 && b >= 0, "Concat node references must be non-negative")
  require(a != b, "Concat requires two distinct nodes")
  require(axis == 0, "Concat supports axis 0 only (sequential juxtaposition)")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}
