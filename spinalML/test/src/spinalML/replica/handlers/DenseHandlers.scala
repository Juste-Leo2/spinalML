// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import spinalML.nn.{BatchNorm1D, LayerNorm1D, Linear}
import spinalML.replica.HWArithmetic._
import spinalML.replica.WeightMemoryLayout.LayerWeightInfo
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

object DenseHandlers {

  def evalLinear(
    l: Linear,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    val inFeatures = l.inFeatures
    val outFeatures = l.outFeatures
    val lanes = l.effLanes
    val nextShape = Seq(1, outFeatures)
    // S2 spill v1: the layout tool emits spilling layers slice-transposed
    // (WeightMemoryLayout R1); gather logical [o][k] rows from the physical
    // `p*Ks*N + n*Ks + k_local` order and fold passes in the replica.
    // Non-spilling layers keep the legacy direct slicing, untouched.
    require((l.spilling && wInfo.spillKSlice > 0) || (!l.spilling && wInfo.spillKSlice <= 0),
      s"DenseHandlers: Linear(spillKSlice=${l.spillKSlice}) vs layout spillKSlice=${wInfo.spillKSlice} mismatch — " +
        "rebuild weights with WeightMemoryLayout.buildDeterministicWeights (no double transposition)")
    val ks = if (wInfo.spillKSlice > 0) wInfo.spillKSlice else inFeatures
    def physIdx(o: Int, k: Int): Int = {
      val p = k / ks
      val kl = k % ks
      p * (ks * outFeatures) + o * ks + kl
    }

    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        val fcW = if (wInfo.spillKSlice <= 0) {
          (0 until outFeatures).map(o => wInfo.weightValues.slice(o * inFeatures, (o + 1) * inFeatures))
        } else {
          (0 until outFeatures).map(o => (0 until inFeatures).map(k => wInfo.weightValues(physIdx(o, k))))
        }
        val fcB = (0 until outFeatures).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
        val out = LayerReplicas.linear(ft.asFloats, fcW, fcB, ft.expBits, ft.mantBits, lanes,
          spillKSlice = wInfo.spillKSlice)
        FloatTensor(nextShape, out, ft.expBits, ft.mantBits)

      case it: IntTensor =>
        val raw = it.asInts
        val outBits = if (wInfo.biasDtype != null) wInfo.biasDtype().getBitsWidth else it.bitWidth
        def wrap(v: Long): Long = {
          if (outBits >= 64) v
          else {
            val mask = (1L << outBits) - 1
            val unsigned = v & mask
            if ((unsigned & (1L << (outBits - 1))) != 0) unsigned - (1L << outBits) else unsigned
          }
        }
        val outInts = (0 until outFeatures).map { o =>
          val fcW = if (wInfo.spillKSlice <= 0) {
            wInfo.weightInts.slice(o * inFeatures, (o + 1) * inFeatures)
          } else {
            (0 until inFeatures).map(k => wInfo.weightInts(physIdx(o, k)))
          }
          val fcB = if (o < wInfo.biasInts.length) wInfo.biasInts(o) else 0L
          var acc = fcB
          for (k <- 0 until inFeatures) {
            val v = if (k < raw.length) raw(k) else 0L
            val w = if (k < fcW.length) fcW(k) else 0L
            acc += v * w
          }
          wrap(acc)
        }
        IntTensor(nextShape, outInts, outBits)
    }
    (nextShape, nextTensor)
  }

  def evalBatchNorm1D(
    bn: BatchNorm1D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    val feat = bn.features
    val ft = curTensor.asInstanceOf[FloatTensor]
    val gamma = wInfo.weightValues.take(feat)
    val beta = wInfo.biasValues.take(feat)
    val out = LayerReplicas.batchNorm1D(ft.asFloats, gamma, beta, ft.expBits, ft.mantBits, feat)
    (curShape, FloatTensor(curShape, out, ft.expBits, ft.mantBits))
  }

  def evalLayerNorm1D(
    ln: LayerNorm1D,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    val feat = ln.features
    val ft = curTensor.asInstanceOf[FloatTensor]
    val gamma = wInfo.weightValues.take(feat)
    val beta = wInfo.biasValues.take(feat)
    val out = LayerReplicas.layerNorm1D(ft.asFloats, feat, gamma, beta, ft.expBits, ft.mantBits)
    (curShape, FloatTensor(curShape, out, ft.expBits, ft.mantBits))
  }
}
