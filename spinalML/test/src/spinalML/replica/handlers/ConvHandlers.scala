// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import scala.collection.mutable.ArrayBuffer
import spinalML.nn.{Conv1D, Conv2D}
import spinalML.replica.HWArithmetic._
import spinalML.replica.WeightMemoryLayout.LayerWeightInfo
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object ConvHandlers {

  def evalConv2D(
    c: Conv2D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    require(curShape.length >= 2, "Conv2D requires 2D or 3D input shape")
    val h = curShape(0)
    val w = curShape(1)
    val inC = if (curShape.length >= 3) curShape(2) else 1
    val kSize = c.kernelSize
    val outC = c.outChannels
    val kElems = kSize * kSize * inC
    val hOut = h - kSize + 1
    val wOut = w - kSize + 1
    val nextShape = Seq(hOut, wOut, outC)

    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor =>
        val arr3D = Array.ofDim[Long](inC, h, w)
        var idx = 0
        val raw = it.asInts
        for (y <- 0 until h; x <- 0 until w; ch <- 0 until inC) {
          arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else 0L
          idx += 1
        }
        val convW = (0 until outC).map(o => wInfo.weightInts.slice(o * kElems, (o + 1) * kElems))
        val convB = (0 until outC).map(o => if (o < wInfo.biasInts.length) wInfo.biasInts(o) else 0L)
        val convOut = LayerReplicas.conv2DInt(arr3D, convW, convB, inC, outC, kSize)
        val flat = LayerReplicas.flattenInt(convOut)
        val outWidth = if (wInfo.biasDtype != null) wInfo.biasDtype().getBitsWidth else 16
        IntTensor(nextShape, flat, outWidth)

      case ft: FloatTensor =>
        val arr3D = Array.ofDim[F](inC, h, w)
        var idx = 0
        val raw = ft.asFloats
        for (y <- 0 until h; x <- 0 until w; ch <- 0 until inC) {
          arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else PZERO
          idx += 1
        }
        val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
        val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
        val convOut = LayerReplicas.conv2D(arr3D, convW, convB, inC, outC, kSize, ft.expBits, ft.mantBits)
        val flat = LayerReplicas.flatten(convOut)
        FloatTensor(nextShape, flat, ft.expBits, ft.mantBits)
    }
    (nextShape, nextTensor)
  }

  def evalConv1D(
    c: Conv1D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    val l = curShape(0)
    val inC = if (curShape.length >= 2) curShape(1) else 1
    val kSize = c.kernelSize
    val outC = c.outChannels
    val kElems = kSize * inC
    val nextShape = Seq(l - kSize + 1, outC)

    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        val arr2D = Array.ofDim[F](l, inC)
        var idx = 0
        val raw = ft.asFloats
        for (pos <- 0 until l; ch <- 0 until inC) {
          arr2D(pos)(ch) = if (idx < raw.length) raw(idx) else PZERO
          idx += 1
        }
        val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
        val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
        val convOut = LayerReplicas.conv1D(arr2D, convW, convB, inC, outC, kSize, ft.expBits, ft.mantBits)
        val flat = ArrayBuffer[F]()
        for (pos <- convOut.indices; ch <- 0 until outC) flat += convOut(pos)(ch)
        FloatTensor(nextShape, flat.toSeq, ft.expBits, ft.mantBits)
      case _: IntTensor =>
        throw new UnsupportedOperationException("Conv1D in int domain not supported")
    }
    (nextShape, nextTensor)
  }
}
