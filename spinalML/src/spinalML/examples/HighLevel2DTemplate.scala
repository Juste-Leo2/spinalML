package spinalML.examples

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * High-Level 2D CNN Template for SpinalML
 *
 * Demonstrates a full 2D vision pipeline with the PyTorch-like `Sequential` /
 * `LayerSpec` API: 2D convolution, requantization, 2D pooling (max + average)
 * and a dense head. All DMA, double-buffering, repacking and stream wiring are
 * generated automatically.
 *
 * Topology and shapes:
 *   [8, 8, 1] -> Conv2D(1->4, K3) [6, 6, 4]   (computed in I32 to avoid overflow)
 *             -> Requantize(I8)
 *             -> ReLU
 *             -> MaxPool2D(2, 2)     [3, 3, 4]
 *             -> AvgPool2D(2, 2)     [1, 1, 4]
 *             -> Flatten             [4, 1]
 *             -> Linear(4->10)       [10, 1]  (computed in I32)
 *             -> Requantize(I8)
 */
case class HighLevel2DTemplate(override val axiConfig: Axi4Config) extends Accelerator(
  dataType = I8(),            // Global quantization format for the network
  inputShape = Seq(8, 8, 1),  // Input image [H, W, C]

  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    // 1. Compute Conv2D in I32 to prevent overflow
    Conv2D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I32())),

    // 2. Requantize the I32 output back to I8 for the rest of the network
    Requantize(shift = 4, targetType = I8()),

    ReLU(),
    MaxPool2D(poolSize = 2, stride = 2),
    AvgPool2D(poolSize = 2, stride = 2),
    Flatten(),

    // 3. Do the same for the final dense layer
    Linear(inFeatures = 4, outFeatures = 10, customType = Some(I32())),
    Requantize(shift = 4, targetType = I8())
  ),

  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object HighLevel2DTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(HighLevel2DTemplate(axiConfig))
}
