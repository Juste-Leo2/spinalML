// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import spinalML.nn.{Cast, Flatten, Repack, Requantize}
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object TransformHandlers {

  def evalCast(
    c: Cast,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val targetData = c.targetType()
    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor =>
        if (targetData.isInstanceOf[spinalML.dtypes.FloatML]) {
          val fType = targetData.asInstanceOf[spinalML.dtypes.FloatML]
          val tExp = fType.expBits
          val tMant = fType.mantBits
          val converted = LayerReplicas.castIntToFloat(it.asInts, it.bitWidth, tExp, tMant, c.scales)
          FloatTensor(curShape, converted, tExp, tMant)
        } else {
          IntTensor(curShape, it.asInts, targetData.getBitsWidth)
        }
      case ft: FloatTensor =>
        if (targetData.isInstanceOf[spinalML.dtypes.FloatML]) {
          val fType = targetData.asInstanceOf[spinalML.dtypes.FloatML]
          val tExp = fType.expBits
          val tMant = fType.mantBits
          val scale = if (c.scales.nonEmpty) c.scales.head else 1.0
          val converted = LayerReplicas.cast(ft.asFloats, scale, ft.expBits, ft.mantBits, tExp, tMant)
          FloatTensor(curShape, converted, tExp, tMant)
        } else {
          throw new UnsupportedOperationException("Cast from Float to Int not supported in replica")
        }
    }
    (curShape, nextTensor)
  }

  def evalFlatten(
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextShape = Seq(1, curShape.product)
    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor   => IntTensor(nextShape, it.asInts, it.bitWidth)
      case ft: FloatTensor => FloatTensor(nextShape, ft.asFloats, ft.expBits, ft.mantBits)
    }
    (nextShape, nextTensor)
  }

  def evalRequantize(
    rq: Requantize,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val outBits = rq.targetType().getBitsWidth
    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor =>
        val req = LayerReplicas.requantizeInt(it.asInts, rq.shift, outBits)
        IntTensor(curShape, req, outBits)
      case _ =>
        throw new UnsupportedOperationException("Requantize is supported in integer domain only")
    }
    (curShape, nextTensor)
  }

  def evalRepack(
    rp: Repack,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    (curShape, curTensor)
  }
}
