package spinalML.examples

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * High-Level API Template for SpinalML
 * 
 * This template demonstrates how to use the PyTorch-like `Sequential` and `LayerSpec` API 
 * to automatically generate a complete neural network hardware accelerator with built-in AXI4 
 * DMA management. You do not need to manually route streams, calculate dimensions, or build DMAs!
 */
case class HighLevelTemplate(override val axiConfig: Axi4Config) extends Accelerator(
  dataType = I16(),            // Global quantization format for the network
  inputShape = Seq(28, 1),    // The expected shape of the input tensor (e.g. 1D signal of length 28)
  
  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    Conv1D(inChannels = 1, outChannels = 4, kernelSize = 3),
    ReLU(),
    MaxPool1D(poolSize = 2, stride = 2),
    Flatten(),
    Linear(inFeatures = 52, outFeatures = 10)
  ),
  
  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object HighLevelTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(HighLevelTemplate(axiConfig))
}
