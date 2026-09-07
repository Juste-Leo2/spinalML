// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalPoolNormDemo
 * Covers the last ops absent from the universal demos:
 * Conv1D -> MaxPool1D -> Cast(int->FP8) -> BatchNorm1D -> LayerNorm1D
 *   -> Flatten -> 2x Linear (fan-out) -> Concat(axis 0) -> [2, 2] FP8.
 */
case class UniversalPoolNormDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(8, 2),
  modelSpec = Seq(
    Conv1D(inChannels = 2, outChannels = 2, kernelSize = 3, customType = Some(I16())), // 0: 8x2 -> 6x2 (I16)
    MaxPool1D(poolSize = 2, stride = 2),                                              // 1: 3x2 I16
    Cast(targetType = FP8_E4M3()),                                                    // 2: int -> float
    BatchNorm1D(features = 2),                                                        // 3: float
    LayerNorm1D(features = 2),                                                        // 4: float
    Flatten(),                                                                        // 5: 1x6
    Linear(inFeatures = 6, outFeatures = 2, customType = Some(FP8_E4M3()),
      customWeightType = Some(FP8_E4M3())),                                           // 6: node 7
    Linear(inFeatures = 2, outFeatures = 2, customType = Some(FP8_E4M3()),
      customWeightType = Some(FP8_E4M3())),                                           // 7: node 8
    Concat(a = 7, b = 8)                                                              // 8: [2, 2] FP8
  ),
  axiConfig = axiConfig
)

import spinalML.io.UartSoC

object UniversalPoolNormDemoVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(new UartSoC(() => UniversalPoolNormDemo(axiConfig), axiConfig = axiConfig, outCount = 4))
}
