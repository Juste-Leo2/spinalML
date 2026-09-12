// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalMultiLane4Demo
 * Demonstrates and validates end-to-end 4-lane parallel acceleration (lanes = 4):
 * Input (I8) -> Conv2D (lanes = 4) -> ReLU -> AvgPool2D (lanes = 4) -> Flatten -> Linear (lanes = 4).
 */
case class UniversalMultiLane4Demo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I16()), lanes = 4),
    ReLU(),
    AvgPool2D(poolSize = 2, stride = 2, lanes = 4),
    Flatten(),
    Linear(inFeatures = 16, outFeatures = 4, customType = Some(I16()), lanes = 4)
  ),
  axiConfig = axiConfig
)
