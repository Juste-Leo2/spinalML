// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.attention.ClassicalAttention
import spinalML.dtypes._

/**
 * High-Level Attention Template for SpinalML
 *
 * Demonstrates a Transformer-style block with the PyTorch-like `Sequential` /
 * `LayerSpec` API using weight-only quantization (wXaY): the activations stay
 * in the float domain (BF16) while the weights are stored as SInt (I8) plus
 * compile-time scales (per-tensor or per-channel), dequantized inside the
 * layers before the float matmuls.
 *
 * Topology and shapes:
 *   [4, 8]  -> Multi-Head ClassicalAttention (4 heads, embedDim=8)  [4, 8]
 *              weights: Wq|Wk|Wv|Wo stacked [32, 8] as I8 + per-channel scales
 *           -> Flatten                                     [32, 1]
 *           -> Linear (32 -> 4), I8 weights + per-tensor scale   [4, 1]
 *
 * All DMA, double-buffering, weight slicing/forking and stream wiring are
 * generated automatically.
 */
case class HighLevelAttentionTemplate(override val axiConfig: Axi4Config) extends Accelerator(
  dataType = BF16(),          // Activations stay in the float domain (softmax policy)
  inputShape = Seq(4, 8),     // [seqLen, embedDim]

  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    // Multi-Head Attention (numHeads must be a power of 2, embedDim % numHeads == 0)
    ClassicalAttention(
      embedDim = 8,
      numHeads = 4,
      customWeightType = Some(I8()),
      weightScales = Seq(0.5, 1.0, 1.5, 2.0, 0.25, 0.75, 1.25, 1.75) // per-channel
    ),

    Flatten(),

    // Classification head, w8a16
    Linear(
      inFeatures = 32,
      outFeatures = 4,
      customWeightType = Some(I8()),
      weightScales = Seq(0.2) // per-tensor
    )
  ),

  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object HighLevelAttentionTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(HighLevelAttentionTemplate(axiConfig))
}
