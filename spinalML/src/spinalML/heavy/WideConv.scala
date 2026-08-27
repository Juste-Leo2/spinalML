package spinalML.heavy

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * Phase-3 D4 validation model: WIDE image (64x64) single-channel classifier.
 *
 * Purpose: exercise the vertical band tiling on a genuinely large spatial
 * input (4 bands of 16 rows vs the 28-row MNIST which fits mostly whole).
 * Topology stays minimal; weights are deterministic pseudo-random values
 * (seeded generator), shared between the hardware packer and the JVM
 * replica, so no training is involved.
 *
 * Conv 3x3 (1->1) -> ReLU -> MaxPool 2x2 -> Flatten -> Linear 961->10.
 * The LogSoftmax head is omitted (argmax(logits) argmax(softmax(logits))).
 *
 * `side` shrinks the square input (the FC size derives from it: fc =
 * ((side-2)/2)^2) so the gate can run as a small, fast debug configuration:
 * `WideConv(axiConfig, tileHeight = 16, side = 16)`.
 */
case class WideConv(override val axiConfig: Axi4Config, override val tileHeight: Int = -1, side: Int = 64) extends Accelerator(
  dataType    = BF16(),
  inputShape  = Seq(side, side, 1),
  modelSpec   = Seq(
    Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3),
    ReLU(),
    MaxPool2D(poolSize = 2, stride = 2),
    Flatten(),
    Linear(inFeatures = ((side - 2) / 2) * ((side - 2) / 2), outFeatures = 10)
  ),
  axiConfig   = axiConfig,
  weightResidencyCSR = true,
  tileHeight  = tileHeight
)
