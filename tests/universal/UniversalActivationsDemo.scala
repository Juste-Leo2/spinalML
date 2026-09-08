// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalActivationsDemo
 * Exercises ops never covered by the other universal demos:
 * Conv2D -> MaxPool2D -> Cast(int->FP8) -> LeakyReLU -> Sigmoid -> Tanh
 *   -> Flatten -> Linear (FP8). Validates pooling, the float activation
 *   family and the int->float boundary bind-exact end-to-end.
 */
case class UniversalActivationsDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(8, 8, 1),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, customType = Some(I16())), // 0: 8x8x1 -> 6x6x2 (I16)
    MaxPool2D(poolSize = 2, stride = 2),                                              // 1: 6x6x2 -> 3x3x2 (I16)
    Cast(targetType = FP8_E4M3()),                                                    // 2: int -> float
    LeakyReLU(),                                                                      // 3: float domain
    Sigmoid(),                                                                        // 4
    Tanh(),                                                                           // 5
    Flatten(),                                                                        // 6: 1x18
    Linear(inFeatures = 18, outFeatures = 4, customType = Some(FP8_E4M3()),
      customWeightType = Some(FP8_E4M3()))                                            // 7: out FP8
  ),
  axiConfig = axiConfig
)

