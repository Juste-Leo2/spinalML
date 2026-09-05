package spinalML.replica.handlers

import scala.collection.mutable.ArrayBuffer
import spinalML.nn.{MaxPool1D, MaxPool2D}
import spinalML.replica.HWArithmetic._
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object PoolHandlers {

  def evalMaxPool2D(
    p: MaxPool2D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val h = curShape(0); val w = curShape(1); val c = if (curShape.length >= 3) curShape(2) else 1
    val hOut = (h - p.poolSize) / p.stride + 1
    val wOut = (w - p.poolSize) / p.stride + 1
    val nextShape = Seq(hOut, wOut, c)

    val nextTensor: ReplicaTensor = curTensor match {
      case it: IntTensor =>
        val arr3D = Array.ofDim[Long](c, h, w)
        var idx = 0
        val raw = it.asInts
        for (y <- 0 until h; x <- 0 until w; ch <- 0 until c) {
          arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else 0L
          idx += 1
        }
        val pooled = LayerReplicas.maxPool2DInt(arr3D, p.poolSize, p.stride)
        val flat = LayerReplicas.flattenInt(pooled)
        IntTensor(nextShape, flat, it.bitWidth)

      case ft: FloatTensor =>
        val arr3D = Array.ofDim[F](c, h, w)
        var idx = 0
        val raw = ft.asFloats
        for (y <- 0 until h; x <- 0 until w; ch <- 0 until c) {
          arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else PZERO
          idx += 1
        }
        val pooled = LayerReplicas.maxPool2D(arr3D, p.poolSize, p.stride, ft.expBits, ft.mantBits)
        val flat = LayerReplicas.flatten(pooled)
        FloatTensor(nextShape, flat, ft.expBits, ft.mantBits)
    }
    (nextShape, nextTensor)
  }

  def evalMaxPool1D(
    p: MaxPool1D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int]
  ): (Seq[Int], ReplicaTensor) = {
    val l = curShape(0); val c = if (curShape.length >= 2) curShape(1) else 1
    val lOut = (l - p.poolSize) / p.stride + 1
    val nextShape = Seq(lOut, c)
    val ft = curTensor.asInstanceOf[FloatTensor]
    val arr2D = Array.ofDim[F](l, c)
    var idx = 0
    for (pos <- 0 until l; ch <- 0 until c) {
      arr2D(pos)(ch) = if (idx < ft.asFloats.length) ft.asFloats(idx) else PZERO
      idx += 1
    }
    val pooled = LayerReplicas.maxPool1D(arr2D, p.poolSize, p.stride, ft.expBits, ft.mantBits)
    val flat = ArrayBuffer[F]()
    for (pos <- pooled.indices; ch <- 0 until c) flat += pooled(pos)(ch)
    (nextShape, FloatTensor(nextShape, flat.toSeq, ft.expBits, ft.mantBits))
  }
}
