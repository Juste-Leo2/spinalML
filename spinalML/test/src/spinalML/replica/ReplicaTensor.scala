// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import HWArithmetic._

/**
 * Universal data representations for ModelReplica execution.
 */
sealed trait ReplicaTensor {
  def shape: Seq[Int]
  def toDoubles: Seq[Double]
  def asInts: Seq[Long]
  def asFloats: Seq[F]
  def length: Int
}

case class IntTensor(shape: Seq[Int], data: Seq[Long], bitWidth: Int) extends ReplicaTensor {
  def toDoubles: Seq[Double] = data.map(_.toDouble)
  def asInts: Seq[Long] = data
  def asFloats: Seq[F] = throw new UnsupportedOperationException("IntTensor is not a float tensor")
  def length: Int = data.length
}

case class FloatTensor(shape: Seq[Int], data: Seq[F], expBits: Int, mantBits: Int) extends ReplicaTensor {
  def toDoubles: Seq[Double] = data.map(f => decode(f, expBits, mantBits))
  def asInts: Seq[Long] = throw new UnsupportedOperationException("FloatTensor is not an int tensor")
  def asFloats: Seq[F] = data
  def length: Int = data.length
}

case class LayerExecutionTrace(
  layerIdx: Int,
  layerName: String,
  outShape: Seq[Int],
  first3: Seq[Double],
  last3: Seq[Double],
  minVal: Double,
  maxVal: Double,
  tensor: Seq[F] = Nil,
  replicaTensor: Option[ReplicaTensor] = None
)

case class ForwardResult(
  logits: Seq[Double],
  traces: Seq[LayerExecutionTrace]
)
