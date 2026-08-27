package spinalML.heavy

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * Phase-3 S4 roadmap gate: two-plus-tile chain WITH a skip connection.
 *
 * WideConv's 64x64 image, but with a ResNet-style branch:
 *   node0 = image (64x64)
 *   n1 = Conv 3x3 (1->1)  -> 62x62         ("convK3")
 *   n2 = ReLU                                ("relu")
 *   n3 = Conv 1x1           -> 62x62         ("convK1")
 *   n4 = Add(n2, n3)                         <- SKIP: node2 has TWO consumers
 *   n5 = MaxPool 2x2       -> 31x31
 *   n6 = Flatten           -> 961
 *   n7 = Linear 961->10
 *
 * The builder forks node 2 (immediate consumer convK1 + deferred Add) through
 * the TapBuffer path; with tileHeight 16 the image arrives in 4 bands, so the
 * skip tap (an exact-capacity tensor FIFO) coexists with band seams — the
 * gate checks tile-boundary continuity AND the deferred-branch exactness in
 * one differential against the JVM replica.
 */
case class WideResidual(override val axiConfig: Axi4Config, override val tileHeight: Int = -1) extends Accelerator(
  dataType    = BF16(),
  inputShape  = Seq(64, 64, 1),
  modelSpec   = Seq(
    Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3),
    ReLU(),
    Conv2D(inChannels = 1, outChannels = 1, kernelSize = 1),
    Add(a = 2, b = 3),
    MaxPool2D(poolSize = 2, stride = 2),
    Flatten(),
    Linear(inFeatures = 961, outFeatures = 10)
  ),
  axiConfig   = axiConfig,
  weightResidencyCSR = true,
  tileHeight  = tileHeight
)
