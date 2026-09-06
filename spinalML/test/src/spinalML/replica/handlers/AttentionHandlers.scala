// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica.handlers

import spinalML.attention.ClassicalAttention
import spinalML.replica.HWArithmetic._
import spinalML.replica.WeightMemoryLayout.LayerWeightInfo
import spinalML.replica.{FloatTensor, IntTensor, LayerReplicas, ReplicaTensor}

/**
 * Bit-exact universal replica of ClassicalAttention (prefill V1).
 *
 * Mirrors ClassicalAttentionHW exactly: full Q/K/V projections, per-head
 * column slices, Q*K^T per head, per-row softmax (exact integer block-float
 * partial-sum), probs*V per head, axis-1 head concatenation, final Wo.
 */
object AttentionHandlers {

  /**
   * KV-cache slot for one attention head.
   * KV-cache transition point: prefill builds it once from the complete K/V
   * outputs; the future decoding stage appends one token row per generation
   * step and drives the SAME attentionHead core (the exact integer softmax
   * partial-sum is incremental by construction, a causal mask only zeroes
   * not-yet-visible key rows).
   */
  case class HeadKV(k: Seq[Seq[F]], v: Seq[Seq[F]])

  /**
   * Per-head attention core: Q_h * K_h^T -> row softmax -> probs * V_h.
   * maskForRow(i) (optional, null = prefill) zeroes the contribution of key
   * row j for query row i (future causal/decoding path).
   */
  def attentionHead(
    qHead: Seq[Seq[F]],
    kv: HeadKV,
    e: Int,
    m: Int,
    maskForRow: Int => Seq[Boolean] = null
  ): Seq[Seq[F]] = {
    // Q * K^T: each key row acts as one weight row (out = key index).
    // HW: this matmul runs lanes=1 (q/k fifos are 1-lane) -> MatmulOp chunks
    // K by `lanes` and accumulates the chunk trees SEQUENTIALLY; with lanes=1
    // the result is a strict per-k sequential FP8 accumulation.
    val scoresSeq = LayerReplicas.linear(qHead.flatten, kv.k, Seq.fill(kv.k.length)(PZERO), e, m, 1)
    val scores = scoresSeq.grouped(kv.k.length).toSeq

    // Per-row softmax over the key axis (exact int-block partial-sum core)
    val probs = scores.zipWithIndex.map { case (row, i) =>
      LayerReplicas.softmaxMaskedValues(row, e, m, if (maskForRow == null) null else maskForRow(i))
    }

    // probs * V: weight rows = headDim columns of V (row (c) = V[:, c]).
    // HW: this matmul runs at seqLen lanes (repacked outputs) -> K = seqLen
    // fits one chunk -> a single adder tree (K >= lanes).
    val vT = (0 until kv.v.head.length).map(c => kv.v.map(r => r(c)))
    val ctxSeq = LayerReplicas.linear(probs.flatten, vT, Seq.fill(vT.length)(PZERO), e, m, kv.k.length)
    ctxSeq.grouped(kv.v.head.length).toSeq
  }

  def evalClassicalAttention(
    a: ClassicalAttention,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: LayerWeightInfo
  ): (Seq[Int], ReplicaTensor) = {
    val (seqLen, embedDim) = curShape match {
      case Seq(s, e) => (s, e)
      case other     => throw new UnsupportedOperationException(s"evalClassicalAttention: unsupported shape $other (expected Seq(seqLen, embedDim))")
    }
    require(embedDim == a.embedDim,
      s"evalClassicalAttention: layer embedDim ${a.embedDim} != tensor embedDim $embedDim")
    val numHeads = a.numHeads
    val headDim = embedDim / numHeads
    require(embedDim % numHeads == 0, s"evalClassicalAttention: embedDim $embedDim not divisible by numHeads $numHeads")

    val nextTensor: ReplicaTensor = curTensor match {
      case ft: FloatTensor =>
        val e = ft.expBits
        val m = ft.mantBits

        val xRows = ft.asFloats.grouped(embedDim).toSeq
        require(xRows.length == seqLen, s"evalClassicalAttention: tensor length ${ft.asFloats.length} != seqLen*embedDim ${seqLen * embedDim}")

        // Wq/Wk/Wv/Wo stored sequentially as [4*embedDim, embedDim] (rows of the tile)
        def matrix(offRows: Int): Seq[Seq[F]] =
          (0 until embedDim).map(r =>
            wInfo.weightValues.slice((offRows + r) * embedDim, (offRows + r + 1) * embedDim))
        val wq = matrix(0)
        val wk = matrix(embedDim)
        val wv = matrix(2 * embedDim)
        val wo = matrix(3 * embedDim)

        // 1. Full projections (mirrors HW: projections then per-head column slices).
        // HW: the projection MatmulOp is 1-lane (projLanes = 1) -> sequential over K.
        def project(mat: Seq[Seq[F]]): Seq[F] =
          LayerReplicas.linear(ft.asFloats, mat, Seq.fill(embedDim)(PZERO), e, m, 1)
        val qRaw = project(wq)
        val kRaw = project(wk)
        val vRaw = project(wv)

        // 2-6. Per head: slices -> scores -> softmax -> context
        val contexts: Seq[Seq[Seq[F]]] = (for (h <- 0 until numHeads) yield {
          def headCols(m: Seq[F]): Seq[Seq[F]] =
            m.grouped(embedDim).map(_.slice(headDim * h, headDim * (h + 1))).toSeq
          val kv = HeadKV(headCols(kRaw), headCols(vRaw))
          attentionHead(headCols(qRaw), kv, e, m)
        }).toSeq

        // 7. Concat heads along axis 1, then output projection * Wo.
        // HW: the final MatmulOp runs at wLanes = embedDim -> single-chunk tree.
        val contextRows: Seq[Seq[F]] = (0 until seqLen).map(r => contexts.flatMap(_.apply(r)))
        val yFlat = LayerReplicas.linear(contextRows.flatten, wo, Seq.fill(embedDim)(PZERO), e, m, embedDim)
        FloatTensor(Seq(seqLen, embedDim), yFlat, e, m)

      case _: IntTensor =>
        throw new UnsupportedOperationException("evalClassicalAttention: int-domain attention not supported (V1 attention is float-domain)")
    }
    (Seq(seqLen, embedDim), nextTensor)
  }
}
