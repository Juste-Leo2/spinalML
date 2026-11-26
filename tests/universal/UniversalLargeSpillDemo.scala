// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalLargeSpillDemo
 *
 * Multi-layer network with multi-beat Linear spill (N=16 > 8, P=4 passes) for
 * the `spinalml test` (replica) path:
 * Input [8,8,2] (I8) -> Conv2D (4ch, 3x3) -> ReLU -> AvgPool2D (2x2)
 *   -> Flatten (36) -> Linear (36 -> 16, Ks=12, multi-beat spill) -> ReLU
 *   -> Linear (16 -> 10).
 */
case class UniversalLargeSpillDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(8, 8, 2),
  modelSpec = Seq(
    Conv2D(inChannels = 2, outChannels = 4, kernelSize = 3, customType = Some(I16())),
    ReLU(),
    AvgPool2D(poolSize = 2, stride = 2),
    Requantize(shift = 2, targetType = I8()),
    Flatten(),
    Linear(inFeatures = 36, outFeatures = 16, weightLanes = 4, spillKSlice = 12),
    ReLU(),
    Linear(inFeatures = 16, outFeatures = 10)
  ),
  axiConfig = axiConfig,
  temporal = 1,
  memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(65536))
)
