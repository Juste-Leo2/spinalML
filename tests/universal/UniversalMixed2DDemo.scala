// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalMixed2DDemo
 * Demonstrates 2D mixed-precision hardware acceleration:
 * Input (I8) -> Conv2D (I4 weights, I16 accumulator) -> ReLU (I16)
 *   -> AvgPool2D (I16) -> Cast (to FP8_E4M3) -> Flatten -> Linear (FP8_E4M3).
 */
case class UniversalMixed2DDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3,
      customType = Some(I16()),
      customWeightType = Some(I4())),
    ReLU(),
    AvgPool2D(poolSize = 2, stride = 2),
    Cast(targetType = FP8_E4M3()),
    Flatten(),
    Linear(inFeatures = 8, outFeatures = 2,
      customWeightType = Some(FP8_E4M3()))
  ),
  axiConfig = axiConfig
)

object UniversalMixed2DDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(UniversalMixed2DDemo(axiConfig))
}
