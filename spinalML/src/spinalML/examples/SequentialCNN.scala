package spinalML.examples

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axilite.AxiLite4Config
import spinalML.nn._
import spinalML.tensors.Tensor
import spinalML.dtypes._

/**
 * An example of using the new High-Level Sequential API to generate
 * a hardware neural network accelerator with automated AXI4 memory management.
 * 
 * This top-level component can be synthesized directly onto an FPGA.
 */
case class SequentialCNN(override val axiConfig: Axi4Config) extends Accelerator(
  dataType = I16(),
  inputShape = Seq(8, 8),
  modelSpec = Seq(
    Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3), // 8x8 -> 6x6
    ReLU(),
    Linear(inFeatures = 6 * 6, outFeatures = 1) // 36 features
  ),
  axiConfig = axiConfig
)


// Generate the Verilog for the FPGA
object SequentialCNNVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(SequentialCNN(axiConfig))
}
