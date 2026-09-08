// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.attention.ClassicalAttention
import spinalML.dtypes._

/**
 * UniversalAttentionDemo
 * Demonstrates the ClassicalAttention layer end-to-end in the universal
 * bit-exact engine (prefill, no KV cache in V1):
 * Input (FP8_E4M3) [4 tokens x 8 embedDim] -> Multi-Head Attention
 *   (2 heads x 4 headDim) -> Output [4 x 8].
 * The exact integer block-float softmax partial-sum core is KV-cache ready:
 * a future decoding stage appends one token row per step with a causal mask.
 */
case class UniversalAttentionDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = FP8_E4M3(),
  inputShape = Seq(4, 8),
  modelSpec = Seq(
    ClassicalAttention(embedDim = 8, numHeads = 2,
      customType = Some(FP8_E4M3()),
      customWeightType = Some(FP8_E4M3()))
  ),
  axiConfig = axiConfig
)

