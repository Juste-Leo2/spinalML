// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalOpsDemo
 * Demonstrates and validates newly integrated layers in the Universal Test Engine:
 * Conv2D -> AvgPool2D -> ReLU -> Flatten -> Linear -> Requantize -> Repack.
 */
case class UniversalOpsDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, customType = Some(I16())),
    AvgPool2D(poolSize = 2, stride = 2),
    ReLU(),
    Flatten(),
    Linear(inFeatures = 8, outFeatures = 4, customType = Some(I16())),
    Requantize(shift = 1, targetType = I8()),
    Repack(newLanes = 1)
  ),
  axiConfig = axiConfig
)

object UniversalOpsDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(UniversalOpsDemo(axiConfig))
}
