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

  def evalSigmoid(
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        FloatTensor(curShape, LayerReplicas.sigmoid(ft.asFloats, ft.expBits, ft.mantBits), ft.expBits, ft.mantBits)
      case it: IntTensor =>
        IntTensor(curShape, LayerReplicas.sigmoidInt(it.asInts, it.bitWidth), it.bitWidth)
    }
    (curShape, nextTensor)
  }

  def evalTanh(
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        FloatTensor(curShape, LayerReplicas.tanh(ft.asFloats, ft.expBits, ft.mantBits), ft.expBits, ft.mantBits)
      case it: IntTensor =>
        IntTensor(curShape, LayerReplicas.tanhInt(it.asInts, it.bitWidth), it.bitWidth)
    }
    (curShape, nextTensor)
  }

  def evalSoftmax(
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        val (seqLen, channels) = curShape match {
          case Seq(s, c)   => (s, c)
          case Seq(c)      => (1, c)
          case other       => throw new UnsupportedOperationException(s"evalSoftmax: unsupported shape $other")
        }
        require(ft.asFloats.length == seqLen * channels, s"evalSoftmax: tensor size mismatch ${ft.asFloats.length} vs ${seqLen * channels}")
        val out = ft.asFloats.grouped(channels).flatMap(row => LayerReplicas.softmax(row.toSeq, ft.expBits, ft.mantBits)).toSeq
        FloatTensor(curShape, out, ft.expBits, ft.mantBits)
      case it: IntTensor =>
        throw new UnsupportedOperationException("evalSoftmax: int-domain softmax not yet supported in the universal replica")
    }
    (curShape, nextTensor)
  }
}

