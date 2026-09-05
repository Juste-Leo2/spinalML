// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import spinalML.nn.LeakyReLU
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object ActivationHandlers {

  def evalReLU(
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor =>
        IntTensor(curShape, LayerReplicas.reluInt(it.asInts), it.bitWidth)
      case ft: FloatTensor =>
        FloatTensor(curShape, LayerReplicas.relu1D(ft.asFloats), ft.expBits, ft.mantBits)
    }
    (curShape, nextTensor)
  }

  def evalLeakyReLU(
    lr: LeakyReLU,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        FloatTensor(curShape, LayerReplicas.leakyRelu(ft.asFloats, lr.shift, ft.expBits, ft.mantBits), ft.expBits, ft.mantBits)
      case _: IntTensor =>
        throw new UnsupportedOperationException("LeakyReLU in int domain not supported")
    }
    (curShape, nextTensor)
  }
}
