// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.nn.DagDSL._
import spinalML.attention.ClassicalAttention
import spinalML.dtypes._

/**
 * UniversalTransformerDemo
 * First full prefill transformer block in the universal bit-exact engine:
 *   Attention (2 heads, headDim 4) -> Add(x, Attn(x))    [residual #1]
 *   -> Linear(8->16) -> ReLU -> Linear(16->8) -> Add     [residual #2]
 * Built with the DagDSL residual helper (automatic node indices). FP8.
 */
case class UniversalTransformerDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = FP8_E4M3(),
  inputShape = Seq(4, 8),
  modelSpec =
    residual(0)(Seq(
      ClassicalAttention(embedDim = 8, numHeads = 2,
        customType = Some(FP8_E4M3()),
        customWeightType = Some(FP8_E4M3()))
    )) ++
    residual(2)(Seq(
      Linear(inFeatures = 8, outFeatures = 16,
        customType = Some(FP8_E4M3()),
        customWeightType = Some(FP8_E4M3())),
      ReLU(),
      Linear(inFeatures = 16, outFeatures = 8,
        customType = Some(FP8_E4M3()),
        customWeightType = Some(FP8_E4M3()))
    )),
  axiConfig = axiConfig
)

import spinalML.io.UartSoC

object UniversalTransformerDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(new UartSoC(() => UniversalTransformerDemo(axiConfig), axiConfig = axiConfig, outCount = 32))
}
