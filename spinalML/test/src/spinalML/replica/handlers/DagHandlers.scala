// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import spinalML.nn.{Add, Concat}
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object DagHandlers {

  def evalAdd(
    ad: Add,
    curShape: Seq[Int],
    nodeOutputs: Seq[ReplicaTensor]
  ): (Seq[Int], ReplicaTensor) = {
    val ta = nodeOutputs(ad.a)
    val tb = nodeOutputs(ad.b)
    val nextTensor: ReplicaTensor = (ta, tb) match {
      case (fa: FloatTensor, fb: FloatTensor) =>
        val out = LayerReplicas.add(fa.asFloats, fb.asFloats, fa.expBits, fa.mantBits)
        FloatTensor(curShape, out, fa.expBits, fa.mantBits)
      case (ia: IntTensor, ib: IntTensor) =>
        val out = ia.asInts.zip(ib.asInts).map { case (x, y) => x + y }
        IntTensor(curShape, out, ia.bitWidth)
      case _ =>
        throw new IllegalArgumentException("Cannot Add different tensor types")
    }
    (curShape, nextTensor)
  }

  def evalConcat(
    cc: Concat,
    curShape: Seq[Int],
    nodeOutputs: Seq[ReplicaTensor]
  ): (Seq[Int], ReplicaTensor) = {
    val ta = nodeOutputs(cc.a)
    val tb = nodeOutputs(cc.b)
    val (nextShape, nextTensor): (Seq[Int], ReplicaTensor) = (ta, tb) match {
      case (fa: FloatTensor, fb: FloatTensor) =>
        val out = LayerReplicas.concat(fa.asFloats, fb.asFloats)
        val nShape = Seq(fa.shape.head + fb.shape.head) ++ fa.shape.tail
        (nShape, FloatTensor(nShape, out, fa.expBits, fa.mantBits))
      case (ia: IntTensor, ib: IntTensor) =>
        val out = ia.asInts ++ ib.asInts
        val nShape = Seq(ia.shape.head + ib.shape.head) ++ ia.shape.tail
        (nShape, IntTensor(nShape, out, ia.bitWidth))
      case _ =>
        throw new IllegalArgumentException("Cannot Concat different tensor types")
    }
    (nextShape, nextTensor)
  }
}
