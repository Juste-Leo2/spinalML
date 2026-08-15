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
    // In this basic version, Conv2D flattens the output to (H_out * W_out, 1)
    Seq(hOut * wOut, outChannels)
  }
  
  override def getWeightShape(): Seq[Int] = Seq(kernelSize * kernelSize * inChannels * outChannels, 1)
  override def getBiasShape(): Seq[Int] = Seq(outChannels, 1)
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
  customWeightType: Option[HardType[Data]] = None
) extends LayerSpec {
  override def outType(default: HardType[Data]) = customType.getOrElse(default)
  override def weightType(default: HardType[Data]) = customWeightType.getOrElse(default)
  
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = Seq(outFeatures, 1)
  override def getWeightShape(): Seq[Int] = Seq(inFeatures, 1)
  override def getBiasShape(): Seq[Int] = Seq(1, 1)
}
