package spinalML.examples

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * An example of using the new High-Level Sequential API to generate
 * a hardware neural network accelerator that chains all available 1D operations.
 */
case class Comprehensive1DCNN(override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)) extends Accelerator(
  dataType = I16(),
  inputShape = Seq(16, 1), // Sequence of length 16, 1 channel
  modelSpec = Seq(
    Conv1D(inChannels = 1, outChannels = 1, kernelSize = 3), // L=14, C=1
    BatchNorm1D(features = 1),
    LeakyReLU(shift = 2),
    MaxPool1D(poolSize = 2, stride = 2), // L=7, C=1
    LayerNorm1D(features = 1), // Channels = 1 (pow2)
    Flatten(), // 7 features
    Linear(inFeatures = 7, outFeatures = 1),
    Softmax()
  ),
  axiConfig = axiConfig
)
