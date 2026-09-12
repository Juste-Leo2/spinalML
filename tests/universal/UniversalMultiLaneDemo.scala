// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalMultiLaneDemo
 * Demonstrates and validates end-to-end multi-lane parallel acceleration (lanes = 2):
 * Input (I8) -> Conv2D (lanes = 2) -> ReLU -> AvgPool2D (lanes = 2) -> Flatten -> Linear (lanes = 2).
 */
case class UniversalMultiLaneDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, customType = Some(I16()), lanes = 2),
    ReLU(),
    AvgPool2D(poolSize = 2, stride = 2, lanes = 2),
    Flatten(),
    Linear(inFeatures = 8, outFeatures = 4, customType = Some(I16()), lanes = 2)
  ),
  axiConfig = axiConfig
)
