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
  dataType = I8(),            // Global quantization format for the network
  inputShape = Seq(28, 1),    // The expected shape of the input tensor (e.g. 1D signal of length 28)
  
  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    // 1. Compute Conv1D in I32 to prevent overflow
    Conv1D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I32())),
    
    // 2. Requantize the I32 output back to I8 for the rest of the network
    Requantize(shift = 4, targetType = I8()),
    
    ReLU(),
    MaxPool1D(poolSize = 2, stride = 2),
    Flatten(),
    
    // 3. Do the same for the final dense layer
    Linear(inFeatures = 52, outFeatures = 10, customType = Some(I32())),
    Requantize(shift = 4, targetType = I8())
  ),
  
  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object HighLevelTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(HighLevelTemplate(axiConfig))
}
