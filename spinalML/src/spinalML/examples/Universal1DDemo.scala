// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * Universal1DDemo
 * Demonstrates and validates 1D operations in the Universal Test Engine:
 * Conv1D -> AvgPool1D -> ReLU -> Flatten -> Linear.
 */
case class Universal1DDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(8, 2),
  modelSpec = Seq(
    Conv1D(inChannels = 2, outChannels = 2, kernelSize = 3, customType = Some(I16())), // 8x2 -> 6x2 in I16
    AvgPool1D(poolSize = 2, stride = 2),                                                // 6x2 -> 2x2 in I16
    ReLU(),
    Flatten(),                                                                          // 3x2 -> 1x6 in I16
    Linear(inFeatures = 6, outFeatures = 2, customType = Some(I16())),                  // 1x6 -> 1x2 in I16
    Requantize(shift = 1, targetType = I8())                                            // 1x2 in I8
  ),
  axiConfig = axiConfig
)

object Universal1DDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(Universal1DDemo(axiConfig))
}
