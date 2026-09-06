// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalResidualDemo
 * Demonstrates skip connections in the Universal Bit-Exact Engine:
 * Conv2D -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> Add(skip) -> Requantize.
 * The Add(3, 6) fuses the flat input (node 3, fan-out) with a 3-layer MLP
 * branch, i.e. a multi-operation residual connection: y = x + MLP(x).
 */
case class UniversalResidualDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, customType = Some(I16())), // 0: 6x6x1 -> 4x4x2 (I16)
    ReLU(),                                                                           // 1
    Flatten(),                                                                        // 2: 1x32
    Linear(inFeatures = 32, outFeatures = 8, customType = Some(I16())),               // 3: 1x8
    ReLU(),                                                                           // 4
    Linear(inFeatures = 8, outFeatures = 32, customType = Some(I16())),               // 5: 1x32
    Add(a = 3, b = 6),                                                                // 6: skip y = x + MLP(x) (I16)
    Requantize(shift = 1, targetType = I8())                                          // 7: out I8
  ),
  axiConfig = axiConfig
)

object UniversalResidualDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(UniversalResidualDemo(axiConfig))
}
