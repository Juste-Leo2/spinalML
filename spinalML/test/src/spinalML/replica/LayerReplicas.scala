// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.nn._
import HWArithmetic._

/**
 * Software replicas for all layer operations in SpinalML.
 * Matches RTL arithmetic and bit-level conventions.
 */
object LayerReplicas {

  // --- 2D Convolution ---
  def conv2D(
    input: Array[Array[Array[F]]], // [C_in][H][W]
    weights: Seq[Seq[F]],         // [C_out][K * K * C_in]
    bias: Seq[F],                 // [C_out]
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    expBits: Int,
    mantBits: Int
  ): Array[Array[Array[F]]] = {
    val h = input(0).length
    val w = input(0)(0).length
    val hOut = h - kernelSize + 1
    val wOut = w - kernelSize + 1
    val out = Array.ofDim[F](outChannels, hOut, wOut)

    for (cOut <- 0 until outChannels; y <- 0 until hOut; x <- 0 until wOut) {
      val prods = ArrayBuffer[F]()
      var wIdx = 0
      for (cIn <- 0 until inChannels; r <- 0 until kernelSize; k <- 0 until kernelSize) {
        val pix = input(cIn)(y + r)(x + k)
        val weight = weights(cOut)(wIdx)
        wIdx += 1
        prods += fmul(pix, weight, expBits, mantBits)
      }
      val acc = fadd(PZERO, tree(prods.toSeq, expBits, mantBits), expBits, mantBits)
      out(cOut)(y)(x) = fadd(acc, bias(cOut), expBits, mantBits)
    }
    out
  }

  // --- 1D Convolution ---
  def conv1D(
    input: Array[Array[F]], // [L_in][C_in]
    weights: Seq[Seq[F]],   // [C_out][K * C_in]
    bias: Seq[F],           // [C_out]
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    expBits: Int,
    mantBits: Int
  ): Array[Array[F]] = {
    val l = input.length
    val lOut = l - kernelSize + 1
    val out = Array.ofDim[F](lOut, outChannels)

    for (pos <- 0 until lOut; cOut <- 0 until outChannels) {
      val prods = ArrayBuffer[F]()
      var wIdx = 0
      for (k <- 0 until kernelSize; cIn <- 0 until inChannels) {
        val inVal = input(pos + k)(cIn)
        val wVal = weights(cOut)(wIdx)
        wIdx += 1
        prods += fmul(inVal, wVal, expBits, mantBits)
      }
      val acc = fadd(PZERO, tree(prods.toSeq, expBits, mantBits), expBits, mantBits)
      out(pos)(cOut) = fadd(acc, bias(cOut), expBits, mantBits)
    }
    out
  }

  // --- Activations ---
  def relu(input: Array[Array[Array[F]]]): Array[Array[Array[F]]] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val out = Array.ofDim[F](c, h, w)
    for (i <- 0 until c; y <- 0 until h; x <- 0 until w) {
      val v = input(i)(y)(x)
      out(i)(y)(x) = if (v.s) PZERO else v
    }
    out
  }

  def relu1D(input: Seq[F]): Seq[F] = input.map(v => if (v.s) PZERO else v)

  def leakyRelu(input: Seq[F], shift: Int, expBits: Int, mantBits: Int): Seq[F] = {
    input.map { v =>
      if (!v.s) v
      else {
        val shiftedExp = v.e - shift
        if (shiftedExp <= 0 || v.e == 0) PZERO
        else F(true, shiftedExp, v.m)
      }
    }
  }

  // --- Poolings ---
  def maxPool2D(input: Array[Array[Array[F]]], poolSize: Int, stride: Int, expBits: Int, mantBits: Int): Array[Array[Array[F]]] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    val out = Array.ofDim[F](c, hOut, wOut)

    for (i <- 0 until c; y <- 0 until hOut; x <- 0 until wOut) {
      var maxVal = input(i)(y * stride)(x * stride)
      for (r <- 0 until poolSize; k <- 0 until poolSize) {
        val v = input(i)(y * stride + r)(x * stride + k)
        maxVal = fmax(maxVal, v, expBits, mantBits)
      }
      out(i)(y)(x) = maxVal
    }
    out
  }

  def avgPool2D(input: Array[Array[Array[F]]], poolSize: Int, stride: Int, expBits: Int, mantBits: Int): Array[Array[Array[F]]] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    val out = Array.ofDim[F](c, hOut, wOut)
    val shift = Math.round(Math.log(poolSize * poolSize) / Math.log(2)).toInt

    for (i <- 0 until c; y <- 0 until hOut; x <- 0 until wOut) {
      val nodes = for (r <- 0 until poolSize; k <- 0 until poolSize) yield input(i)(y * stride + r)(x * stride + k)
      val acc = tree(nodes, expBits, mantBits)
      val shiftedExp = acc.e - shift
      out(i)(y)(x) = if (shiftedExp <= 0 || acc.e == 0) PZERO else F(acc.s, shiftedExp, acc.m)
    }
    out
  }

  def maxPool1D(input: Array[Array[F]], poolSize: Int, stride: Int, expBits: Int, mantBits: Int): Array[Array[F]] = {
    val l = input.length; val c = input(0).length
    val lOut = (l - poolSize) / stride + 1
    val out = Array.ofDim[F](lOut, c)

    for (pos <- 0 until lOut; ch <- 0 until c) {
      var maxVal = input(pos * stride)(ch)
      for (k <- 0 until poolSize) {
        maxVal = fmax(maxVal, input(pos * stride + k)(ch), expBits, mantBits)
      }
      out(pos)(ch) = maxVal
    }
    out
  }

  // --- Flatten (features-last [H, W, C]) ---
  def flatten(input: Array[Array[Array[F]]]): Seq[F] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val out = ArrayBuffer[F]()
    for (y <- 0 until h; x <- 0 until w; i <- 0 until c) {
      out += input(i)(y)(x)
    }
    out.toSeq
  }

  // --- Linear / Dense Layer ---
  def linear(
    input: Seq[F],
    weights: Seq[Seq[F]], // [outFeatures][inFeatures]
    bias: Seq[F],         // [outFeatures]
    expBits: Int,
    mantBits: Int,
    weightLanes: Int
  ): Seq[F] = {
    val inFeatures = weights.head.length
    val outFeatures = weights.length
    val rows = input.length / inFeatures
    val out = ArrayBuffer[F]()

    for (r <- 0 until rows) {
      val rowInput = input.slice(r * inFeatures, (r + 1) * inFeatures)
      for (o <- 0 until outFeatures) {
        var acc = PZERO
        for (chunk <- 0 until inFeatures by weightLanes) {
          val len = math.min(weightLanes, inFeatures - chunk)
          val prods = (0 until len).map(k => fmul(rowInput(chunk + k), weights(o)(chunk + k), expBits, mantBits))
          acc = fadd(acc, tree(prods, expBits, mantBits), expBits, mantBits)
        }
        out += fadd(acc, bias(o), expBits, mantBits)
      }
    }
    out.toSeq
  }

  // --- Normalizations ---
  def batchNorm1D(input: Seq[F], gamma: Seq[F], beta: Seq[F], expBits: Int, mantBits: Int): Seq[F] = {
    input.indices.map { i =>
      fadd(fmul(input(i), gamma(i), expBits, mantBits), beta(i), expBits, mantBits)
    }
  }

  // --- Cast / Dequantization ---
  def cast(input: Seq[F], scale: Double, inExp: Int, inMant: Int, outExp: Int, outMant: Int): Seq[F] = {
    val scaleF = fromDouble(scale, outExp, outMant)
    input.map { v =>
      val realVal = decode(v, inExp, inMant)
      val converted = fromDouble(realVal, outExp, outMant)
      if (scale == 1.0) converted else fmul(converted, scaleF, outExp, outMant)
    }
  }

  // --- DAG Merge Operations ---
  def add(a: Seq[F], b: Seq[F], expBits: Int, mantBits: Int): Seq[F] = {
    require(a.length == b.length, "Add inputs must have the same length")
    a.indices.map(i => fadd(a(i), b(i), expBits, mantBits))
  }

  def concat(a: Seq[F], b: Seq[F]): Seq[F] = a ++ b

  // --- Integer Domain Operations ---
  def conv2DInt(
    input: Array[Array[Array[Long]]], // [C_in][H][W]
    weights: Seq[Seq[Long]],          // [C_out][K * K * C_in]
    bias: Seq[Long],                  // [C_out]
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int
  ): Array[Array[Array[Long]]] = {
    val h = input(0).length
    val w = input(0)(0).length
    val hOut = h - kernelSize + 1
    val wOut = w - kernelSize + 1
    val out = Array.ofDim[Long](outChannels, hOut, wOut)

    for (cOut <- 0 until outChannels; y <- 0 until hOut; x <- 0 until wOut) {
      var acc = if (cOut < bias.length) bias(cOut) else 0L
      var wIdx = 0
      for (cIn <- 0 until inChannels; r <- 0 until kernelSize; k <- 0 until kernelSize) {
        val pix = input(cIn)(y + r)(x + k)
        val weight = weights(cOut)(wIdx)
        wIdx += 1
        acc += pix * weight
      }
      out(cOut)(y)(x) = acc
    }
    out
  }

  def reluInt(input: Seq[Long]): Seq[Long] = input.map(v => math.max(v, 0L))

  def maxPool2DInt(
    input: Array[Array[Array[Long]]],
    poolSize: Int,
    stride: Int
  ): Array[Array[Array[Long]]] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    val out = Array.ofDim[Long](c, hOut, wOut)

    for (i <- 0 until c; y <- 0 until hOut; x <- 0 until wOut) {
      var maxVal = input(i)(y * stride)(x * stride)
      for (r <- 0 until poolSize; k <- 0 until poolSize) {
        val v = input(i)(y * stride + r)(x * stride + k)
        maxVal = math.max(maxVal, v)
      }
      out(i)(y)(x) = maxVal
    }
    out
  }

  def flattenInt(input: Array[Array[Array[Long]]]): Seq[Long] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val out = ArrayBuffer[Long]()
    for (y <- 0 until h; x <- 0 until w; i <- 0 until c) {
      out += input(i)(y)(x)
    }
    out.toSeq
  }

  def castIntToFloat(
    input: Seq[Long],
    inWidth: Int,
    outExp: Int,
    outMant: Int,
    scales: Seq[Double]
  ): Seq[F] = {
    val useScale = scales.nonEmpty && !(scales.length == 1 && scales.head == 1.0)
    val scaleLits = if (useScale) scales.map(s => fromDouble(s, outExp, outMant)) else Nil
    input.zipWithIndex.map { case (v, idx) =>
      val converted = fromSInt(v, inWidth, outExp, outMant)
      if (useScale) {
        val scaleLit = if (scaleLits.length == 1) scaleLits.head else scaleLits(idx % scaleLits.length)
        fmul(converted, scaleLit, outExp, outMant)
      } else {
        converted
      }
    }
  }
}
