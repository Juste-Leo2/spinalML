// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package examples.Mnist

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * Tang Primer 20K MNIST Hardware Accelerator.
 *
 * Implements a high-efficiency mixed-precision CNN architecture (W4A8):
 * - Input: 28x28 grayscale image (8-bit signed integer activations)
 * - Conv2D: 1 -> 2 channels, 5x5 kernel in INT4 (I4) with I16 accumulation
 * - ReLU & MaxPool2D (2x2)
 * - Cast to FP8 (E4M3) with dequantization scaling
 * - Flatten (288 features)
 * - Linear: 288 -> 10 output logits in FP8
 *
 * Can be compiled directly to an FPGA bitstream via the SpinalML CLI:
 * {{{
 * python cli/main.py build examples/Mnist/Model.scala --board tang-primer-20k --no-dsp
 * }}}
 */
case class Model(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4),
  override val tileHeight: Int = -1,
  override val temporal: Int = 16
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(28, 28, 1),
  modelSpec = Seq(
    // 1. Convolution 2D with true 4-bit nibble-packed weights (I4)
    Conv2D(
      inChannels = 1, 
      outChannels = 2, 
      kernelSize = 5,
      customType = Some(I16()),
      customWeightType = Some(I4())
    ),
    ReLU(),
    MaxPool2D(poolSize = 2, stride = 2),
    
    // 2. Transition from Integer domain to Floating-Point domain (FP8 E4M3)
    Cast(FP8_E4M3(), scales = Seq(0.08544921875)),
    Flatten(),
    
    // 3. Dense layer mapping to the 10 digit classes (0-9)
    Linear(
      inFeatures = 288, 
      outFeatures = 10,
      customWeightType = Some(FP8_E4M3()), 
      weightLanes = 4
    )
  ),
  axiConfig = axiConfig,
  temporal = temporal
)
