// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.{RoundingConfig, RoundingMode}
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
    mantBits: Int,
    // M2/Phase-4 K-axis chunk width mirroring the HW matmul fold
    // (Conv2D.weightLanes): chunks accumulate sequentially via fadd, exactly
    // like Linear. <= 0 (default) = single chunk over the whole window, the
    // historical behavior. Must divide K*K*inChannels when > 0.
    lanes: Int = -1,
    // P1 spill (docs/ddr_spill_ops.md): K-slice width streamed per HW pass
    // over the flattened window axis. <= 0 (default) = single pass, the
    // historical behavior, instruction by instruction. > 0 = passes fold
    // outermost (seeded from the running acc, bias once at the end), exactly
    // like the multi-pass engine. Must divide the window, and be a multiple
    // of lanes when > 0 (mirrors the LayerSpec spillKSlice requires).
    spillKSlice: Int = -1
  ): Array[Array[Array[F]]] = {
    val h = input(0).length
    val w = input(0)(0).length
    val hOut = h - kernelSize + 1
    val wOut = w - kernelSize + 1
    val out = Array.ofDim[F](outChannels, hOut, wOut)

    for (cOut <- 0 until outChannels; y <- 0 until hOut; x <- 0 until wOut) {
      val prods = ArrayBuffer[F]()
      var wIdx = 0
      // Window order matches the HW im2col shift register ([K,K,C] row-major:
      // kernel row, kernel col, channel fastest) so flat index wIdx is the
      // DDR offset the engine reads. C=1 degenerates to the legacy order.
      for (r <- 0 until kernelSize; k <- 0 until kernelSize; cIn <- 0 until inChannels) {
        val pix = input(cIn)(y + r)(x + k)
        val weight = weights(cOut)(wIdx)
        wIdx += 1
        prods += fmul(pix, weight, expBits, mantBits)
      }
      val wLanes = if (lanes <= 0) prods.length else lanes
      require(prods.length % wLanes == 0,
        s"conv2D: window ${prods.length} must be a multiple of lanes=$wLanes (dense chunks, no padding)")
      var acc = PZERO
      if (spillKSlice <= 0) {
        for (chunk <- 0 until prods.length by wLanes) {
          acc = fadd(acc, tree(prods.slice(chunk, chunk + wLanes).toSeq, expBits, mantBits), expBits, mantBits)
        }
      } else {
        require(prods.length % spillKSlice == 0,
          s"conv2D: window ${prods.length} must be a multiple of spillKSlice=$spillKSlice (dense passes, no padding)")
        require(spillKSlice % wLanes == 0,
          s"conv2D: spillKSlice=$spillKSlice must be a multiple of lanes=$wLanes (pass-internal chunking matches HW)")
        for (p <- 0 until prods.length by spillKSlice) {
          for (chunk <- 0 until spillKSlice by wLanes) {
            val len = math.min(wLanes, spillKSlice - chunk)
            acc = fadd(acc, tree(prods.slice(p + chunk, p + chunk + len).toSeq, expBits, mantBits), expBits, mantBits)
          }
        }
      }
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
    mantBits: Int,
    // Same M2 chunk contract as conv2D (mirrors Conv1D.weightLanes).
    lanes: Int = -1
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
      val wLanes = if (lanes <= 0) prods.length else lanes
      require(prods.length % wLanes == 0,
        s"conv1D: window ${prods.length} must be a multiple of lanes=$wLanes (dense chunks, no padding)")
      var acc = PZERO
      for (chunk <- 0 until prods.length by wLanes) {
        acc = fadd(acc, tree(prods.slice(chunk, chunk + wLanes).toSeq, expBits, mantBits), expBits, mantBits)
      }
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
    weightLanes: Int,
    // S2 spill contract (docs/ddr_final_impl.md): K-slice width streamed per
    // HW pass. <= 0 (default) = single pass over the whole row, the
    // historical behavior, instruction by instruction. > 0 = passes fold
    // outermost (seeded from the running acc, bias once at the end), exactly
    // like the multi-pass GEMM. Must divide inFeatures, and be a multiple of
    // weightLanes when > 0 (mirrors the LayerSpec spillKSlice requires).
    spillKSlice: Int = -1
  ): Seq[F] = {
    val inFeatures = weights.head.length
    val outFeatures = weights.length
    val rows = input.length / inFeatures
    val out = ArrayBuffer[F]()

    if (spillKSlice <= 0) {
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
    } else {
      require(inFeatures % spillKSlice == 0,
        s"linear: inFeatures=$inFeatures must be a multiple of spillKSlice=$spillKSlice (dense passes, no padding)")
      require(spillKSlice % weightLanes == 0,
        s"linear: spillKSlice=$spillKSlice must be a multiple of weightLanes=$weightLanes (pass-internal chunking matches HW)")
      for (r <- 0 until rows) {
        val rowInput = input.slice(r * inFeatures, (r + 1) * inFeatures)
        for (o <- 0 until outFeatures) {
          var acc = PZERO
          for (p <- 0 until inFeatures by spillKSlice) {
            for (chunk <- 0 until spillKSlice by weightLanes) {
              val len = math.min(weightLanes, spillKSlice - chunk)
              val prods = (0 until len).map(k => fmul(rowInput(p + chunk + k), weights(o)(p + chunk + k), expBits, mantBits))
              acc = fadd(acc, tree(prods, expBits, mantBits), expBits, mantBits)
            }
          }
          out += fadd(acc, bias(o), expBits, mantBits)
        }
      }
    }
    out.toSeq
  }

  // --- Normalizations ---
  def batchNorm1D(input: Seq[F], gamma: Seq[F], beta: Seq[F], expBits: Int, mantBits: Int, features: Int = -1): Seq[F] = {
    val feat = if (features > 0) features else gamma.length
    require(feat > 0 && gamma.length % feat == 0 && beta.length % feat == 0,
      s"batchNorm1D: invalid gamma/beta lengths (${gamma.length}, ${beta.length}) for features=$feat")
    input.indices.map { i =>
      val c = i % feat
      fadd(fmul(input(i), gamma(c), expBits, mantBits), beta(c), expBits, mantBits)
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
      // Same (r,k,c) window order as the float path / HW shift register.
      for (r <- 0 until kernelSize; k <- 0 until kernelSize; cIn <- 0 until inChannels) {
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
    scales: Seq[Double],
    rounding: RoundingMode = RoundingConfig.current
  ): Seq[F] = {
    val useScale = scales.nonEmpty && !(scales.length == 1 && scales.head == 1.0)
    val scaleLits = if (useScale) scales.map(s => fromDouble(s, outExp, outMant)) else Nil
    input.zipWithIndex.map { case (v, idx) =>
      val converted = fromSInt(v, inWidth, outExp, outMant, rounding)
      if (useScale) {
        val scaleLit = if (scaleLits.length == 1) scaleLits.head else scaleLits(idx % scaleLits.length)
        fmul(converted, scaleLit, outExp, outMant)
      } else {
        converted
      }
    }
  }

  // --- 1D Average Pooling ---
  def avgPool1D(input: Array[Array[F]], poolSize: Int, stride: Int, expBits: Int, mantBits: Int): Array[Array[F]] = {
    val l = input.length; val c = input(0).length
    val lOut = (l - poolSize) / stride + 1
    val out = Array.ofDim[F](lOut, c)
    val shift = Math.round(Math.log(poolSize) / Math.log(2)).toInt

    for (pos <- 0 until lOut; ch <- 0 until c) {
      val nodes = for (k <- 0 until poolSize) yield input(pos * stride + k)(ch)
      val acc = tree(nodes, expBits, mantBits)
      val shiftedExp = acc.e - shift
      out(pos)(ch) = if (shiftedExp <= 0 || acc.e == 0) PZERO else F(acc.s, shiftedExp, acc.m)
    }
    out
  }

  def avgPool2DInt(input: Array[Array[Array[Long]]], poolSize: Int, stride: Int, outBits: Int,
                   rounding: RoundingMode = RoundingConfig.current): Array[Array[Array[Long]]] = {
    val c = input.length; val h = input(0).length; val w = input(0)(0).length
    val hOut = (h - poolSize) / stride + 1
    val wOut = (w - poolSize) / stride + 1
    val out = Array.ofDim[Long](c, hOut, wOut)
    val shift = Math.round(Math.log(poolSize * poolSize) / Math.log(2)).toInt

    for (i <- 0 until c; y <- 0 until hOut; x <- 0 until wOut) {
      var acc = 0L
      for (r <- 0 until poolSize; k <- 0 until poolSize) {
        acc += input(i)(y * stride + r)(x * stride + k)
      }
      out(i)(y)(x) = requantizeScalar(acc, shift, outBits, rounding)
    }
    out
  }

  def avgPool1DInt(input: Array[Array[Long]], poolSize: Int, stride: Int, outBits: Int,
                   rounding: RoundingMode = RoundingConfig.current): Array[Array[Long]] = {
    val l = input.length; val c = input(0).length
    val lOut = (l - poolSize) / stride + 1
    val out = Array.ofDim[Long](lOut, c)
    val shift = Math.round(Math.log(poolSize) / Math.log(2)).toInt

    for (pos <- 0 until lOut; ch <- 0 until c) {
      var acc = 0L
      for (k <- 0 until poolSize) {
        acc += input(pos * stride + k)(ch)
      }
      out(pos)(ch) = requantizeScalar(acc, shift, outBits, rounding)
    }
    out
  }

  // --- Non-linear Activations (Sigmoid / Tanh) ---
  def sigmoid(input: Seq[F], expBits: Int, mantBits: Int): Seq[F] = {
    val bitWidth = expBits + mantBits + 1
    if (bitWidth <= 8) {
      val valFn = spinalML.utils.MathLUTs.floatValFn(expBits, mantBits)
      val encFn = spinalML.utils.MathLUTs.floatEncodeFn(expBits, mantBits)
      input.map { f =>
        val negF = F(!f.s, f.e, f.m)
        val negBits = (if (negF.s) 1 << (expBits + mantBits) else 0) | (negF.e << mantBits) | negF.m
        val negReal = valFn(negBits)
        val expReal = Math.exp(negReal)
        val expEnc = encFn(expReal).toInt
        val expF = F((expEnc >> (expBits + mantBits) & 1) == 1, (expEnc >> mantBits) & ((1 << expBits) - 1), expEnc & ((1 << mantBits) - 1))
        val oneF = fromDouble(1.0, expBits, mantBits)
        val addF = fadd(expF, oneF, expBits, mantBits)
        val addBits = (if (addF.s) 1 << (expBits + mantBits) else 0) | (addF.e << mantBits) | addF.m
        val addReal = valFn(addBits)
        val recReal = if (addReal == 0.0) 0.0 else 1.0 / addReal
        val recEnc = encFn(recReal).toInt
        F((recEnc >> (expBits + mantBits) & 1) == 1, (recEnc >> mantBits) & ((1 << expBits) - 1), recEnc & ((1 << mantBits) - 1))
      }
    } else {
      input.map { f =>
        val d = decode(f, expBits, mantBits)
        val s = 1.0 / (1.0 + math.exp(-d))
        fromDouble(s, expBits, mantBits)
      }
    }
  }

  /** Two's-complement bits -> signed Long (replica IntTensor convention). */
  private def bitsToSigned(bits: Long, bitWidth: Int): Long =
    if (bits >= (1L << (bitWidth - 1))) bits - (1L << bitWidth) else bits

  /**
   * Quantized logistic, TFLite int8/uint8 convention (out scale 1/256, int8
   * zp -128). Mirrors the RTL `QuantActivation` ROM: same formula, same
   * switch-aware encoding function.
   */
  def sigmoidInt(input: Seq[Long], bitWidth: Int, inputScale: Double = 1.0, inputZeroPoint: Int = 0,
                 rounding: RoundingMode = RoundingConfig.current): Seq[Long] = {
    val encFn = spinalML.utils.MathLUTs.intEncodeFn(bitWidth, rounding)
    input.map { v =>
      val x = (v - inputZeroPoint) * inputScale
      bitsToSigned(encFn(1.0 / (1.0 + math.exp(-x)) * 256.0 - 128.0).toLong, bitWidth)
    }
  }

  def tanh(input: Seq[F], expBits: Int, mantBits: Int): Seq[F] = {
    val two = fromDouble(2.0, expBits, mantBits)
    val minusOne = fromDouble(-1.0, expBits, mantBits)
    val x2 = input.map(f => fmul(f, two, expBits, mantBits))
    val sig = sigmoid(x2, expBits, mantBits)
    sig.map(s => fadd(fmul(s, two, expBits, mantBits), minusOne, expBits, mantBits))
  }

  /**
   * Quantized tanh, TFLite int8/uint8 convention (out scale 1/128, int8 zp 0).
   * Same oracle as the RTL `QuantActivation` tanh ROM.
   */
  def tanhInt(input: Seq[Long], bitWidth: Int, inputScale: Double = 1.0, inputZeroPoint: Int = 0,
              rounding: RoundingMode = RoundingConfig.current): Seq[Long] = {
    val encFn = spinalML.utils.MathLUTs.intEncodeFn(bitWidth, rounding)
    input.map { v =>
      val x = (v - inputZeroPoint) * inputScale
      bitsToSigned(encFn(math.tanh(x) * 128.0).toLong, bitWidth)
    }
  }

  // --- Softmax (mirror of Softmax1D) ---
  /** Software mirror of Softmax1D for narrow floats (bitWidth <= 8, e.g. FP8):
    * max (dtype) -> sub (dtype) -> exp LUT -> exact int block-float sum (fixed
    * reference exponent) -> normalize RNE -> reciprocal LUT -> final mul. */
  def softmax(input: Seq[F], expBits: Int, mantBits: Int): Seq[F] =
    softmaxMaskedValues(input, expBits, mantBits, null)

  /**
   * Softmax over one row with an optional mask (mask(i) == false zeroes the
   * contribution of channel i, i.e. the partial-sum accumulator skips it).
   * Prefill passes mask == null; the KV-cache/causal (decoding) path reuses
   * this core with a triangular mask while the exact integer partial-sum
   * stays valid when key/value rows are appended incrementally.
   */
  def softmaxMaskedValues(input: Seq[F], expBits: Int, mantBits: Int, mask: Seq[Boolean]): Seq[F] = {
    val values = (0 until input.length).map(i => if (mask != null && !mask(i)) PZERO else input(i))
    softmaxCore(values, expBits, mantBits)
  }

  private def softmaxCore(input: Seq[F], expBits: Int, mantBits: Int): Seq[F] = {
    val bitWidth = expBits + mantBits + 1
    require(bitWidth <= 8, "Universal replica softmax currently supports only <= 8-bit floats")
    val valFn = spinalML.utils.MathLUTs.floatValFn(expBits, mantBits)
    val encFn = spinalML.utils.MathLUTs.floatEncodeFn(expBits, mantBits)

    def bitsOf(f: F): Int = (if (f.s) 1 << (expBits + mantBits) else 0) | (f.e << mantBits) | f.m
    def fFromBits(b: Int): F = F((b >> (expBits + mantBits) & 1) == 1, (b >> mantBits) & ((1 << expBits) - 1), b & ((1 << mantBits) - 1))
    def latEncode(real: Double): F = fFromBits(encFn(real).toInt)
    def latDecode(f: F): Double = valFn(bitsOf(f))
    def neg(f: F): F = F(!f.s, f.e, f.m)

    // 1. Max (dtype) in hardware order = simple fold (max is commutative)
    val maxF = input.reduce((a, b) => fmax(a, b, expBits, mantBits))
    val negMax = neg(maxF)

    // 2. Sub + 3. Exp LUT (encode of Math.exp on the LUT-decoded input)
    val shifted = input.map(v => fadd(v, negMax, expBits, mantBits))
    val exps = shifted.map(f => latEncode(Math.exp(latDecode(f))))

    // 4. Exact integer block-float sum (fixed reference: exp field 1 = LSB)
    val ints = exps.map { f =>
      if (f.e == 0) 0L
      else ((1L << mantBits) | f.m) << (f.e - 1)
    }
    val total = ints.sum
    val maxE = (1 << expBits) - 1

    def normalizeBlock(total: Long): F = {
      if (total == 0) return PZERO
      val p = 63 - java.lang.Long.numberOfLeadingZeros(total)
      if (p - mantBits + 1 > maxE) return F(false, maxE, 0)
      val raw = total & ((1L << p) - 1)
      val (mantv0, gbit, stbit) =
        if (p > mantBits) {
          val m0 = (raw >> (p - mantBits)).toInt
          val g0 = ((total >> (p - mantBits - 1)) & 1L) == 1
          val s0 = (total & ((1L << (p - mantBits - 1)) - 1)) != 0
          (m0, g0, s0)
        } else {
          ((raw << (mantBits - p)).toInt, false, false)
        }
      var manti = mantv0 + (if (gbit && ((mantv0 & 1) == 1 || stbit)) 1 else 0)
      var pv = p
      if (manti >= (1 << mantBits)) {
        manti = 0
        pv += 1
      }
      val expF = pv - mantBits + 1
      if (expF > maxE) F(false, maxE, 0)
      else F(false, expF, manti)
    }
    val sumF = normalizeBlock(total)

    // 5. Reciprocal LUT (with ReciprocalOp divide-by-zero guard)
    val sumReal = latDecode(sumF)
    val recReal = 1.0 / (sumReal + (if (sumReal >= 0) 1e-9 else -1e-9))
    val recipF = latEncode(recReal)

    // 6. Final multiply per channel
    exps.map(e => fmul(e, recipF, expBits, mantBits))
  }

  // --- Layer Normalization 1D ---
  def layerNorm1D(
    input: Seq[F],
    channels: Int,
    gamma: Seq[F],
    beta: Seq[F],
    expBits: Int,
    mantBits: Int
  ): Seq[F] = {
    require(input.length % channels == 0, "LayerNorm1D input length must be multiple of channels")
    val logN = Math.round(Math.log(channels) / Math.log(2)).toInt
    val seqLen = input.length / channels
    val out = ArrayBuffer[F]()

    def divN(f: F): F = {
      val expSInt = f.e - logN
      if (f.e == 0 || expSInt <= 0) PZERO
      else F(f.s, expSInt, f.m)
    }

    for (t <- 0 until seqLen) {
      val row = input.slice(t * channels, (t + 1) * channels)
      val sumX = tree(row, expBits, mantBits)
      val mean = divN(sumX)

      val diffs = row.map(x => fadd(x, F(!mean.s, mean.e, mean.m), expBits, mantBits))
      val sqDiffs = diffs.map(d => fmul(d, d, expBits, mantBits))
      val sumSq = tree(sqDiffs, expBits, mantBits)
      val variance = divN(sumSq)

      // LAY-04: same epsilon policy as the RTL. Encoded 1e-5 when
      // representable (BF16), otherwise the smallest positive normal
      // (E4M3 2^-6, E2M1 1.0) — without it, a zero (or underflowed) variance
      // reaches rsqrt at input 0 and saturates the inverse standard deviation.
      val epsF = {
        val enc = fromDouble(1e-5, expBits, mantBits)
        if (enc.e == 0) F(false, 1, 0) else enc
      }
      val varWithEps = fadd(variance, epsF, expBits, mantBits)

      val varDouble = decode(varWithEps, expBits, mantBits)
      val rsqrtVal = if (varDouble <= 0) PZERO else fromDouble(1.0 / math.sqrt(varDouble), expBits, mantBits)

      for (ch <- 0 until channels) {
        val g = if (ch < gamma.length) gamma(ch) else fromDouble(1.0, expBits, mantBits)
        val b = if (ch < beta.length) beta(ch) else PZERO
        val norm = fmul(diffs(ch), rsqrtVal, expBits, mantBits)
        val scaled = fmul(norm, g, expBits, mantBits)
        out += fadd(scaled, b, expBits, mantBits)
      }
    }
    out.toSeq
  }

  // --- Requantize ---
  /**
   * Mirror of RequantizeOp: shift (RNE by default, trunc legacy) then
   * saturation. RNE on an arithmetic shift rounds guard/sticky ties to even;
   * the switch follows [[spinalML.RoundingConfig]] like the elaborated RTL.
   */
  def requantizeScalar(v: Long, shift: Int, outBits: Int,
                       rounding: RoundingMode = RoundingConfig.current): Long = {
    val maxVal = (1L << (outBits - 1)) - 1
    val minVal = -(1L << (outBits - 1))
    val useRne = rounding == RoundingMode.Rne
    val shifted =
      if (shift <= 0) v
      else if (!useRne) v >> shift
      else {
        val truncated = v >> shift
        val rem = v - (truncated << shift)
        val half = 1L << (shift - 1)
        if (rem > half || (rem == half && (truncated & 1L) != 0)) truncated + 1 else truncated
      }
    math.max(minVal, math.min(maxVal, shifted))
  }

  def requantizeInt(input: Seq[Long], shift: Int, outBits: Int,
                    rounding: RoundingMode = RoundingConfig.current): Seq[Long] =
    input.map(requantizeScalar(_, shift, outBits, rounding))
}

