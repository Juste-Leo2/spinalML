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

case class Conv2D(
  inChannels: Int, 
  outChannels: Int, 
  kernelSize: Int,
  customType: Option[HardType[Data]] = None,
  customWeightType: Option[HardType[Data]] = None
) extends LayerSpec {
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
  weightScales: Seq[Double] = Seq(1.0)
) extends LayerSpec {
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
  customWeightType: Option[HardType[Data]] = None
) extends LayerSpec {
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

case class Softmax() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class BatchNorm1D(features: Int) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(features, 1) // gamma
  override def getBiasShape(): Seq[Int] = Seq(features, 1) // beta
}

case class LayerNorm1D(features: Int) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(features, 1) // gamma
  override def getBiasShape(): Seq[Int] = Seq(features, 1) // beta
}

case class MaxPool1D(poolSize: Int, stride: Int) extends LayerSpec {
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

case class AvgPool1D(poolSize: Int, stride: Int) extends LayerSpec {
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

case class MaxPool2D(poolSize: Int, stride: Int) extends LayerSpec {
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

case class AvgPool2D(poolSize: Int, stride: Int) extends LayerSpec {
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

case class Sigmoid() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Tanh() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Cast(targetType: HardType[Data]) extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
  override def outType(default: HardType[Data]) = targetType
}

case class Flatten() extends LayerSpec {
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = Seq(1, inShape.product)
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}

case class Requantize(shift: Int, targetType: HardType[Data]) extends LayerSpec {
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

case class Concat(a: Int, b: Int, axis: Int = 0) extends LayerSpec {
  require(a >= 0 && b >= 0, "Concat node references must be non-negative")
  require(a != b, "Concat requires two distinct nodes")
  require(axis == 0, "Concat supports axis 0 only (sequential juxtaposition)")
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = inShape
  override def getWeightShape(): Seq[Int] = Seq(0)
  override def getBiasShape(): Seq[Int] = Seq(0)
}
