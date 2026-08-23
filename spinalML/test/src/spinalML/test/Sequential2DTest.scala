package spinalML.test

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes.{I8, I32, BF16}

class Sequential2DTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  test("Sequential 2D CNN with MaxPool2D/AvgPool2D compilation") {
    // [8,8,1] -> Conv2D(1->4,K3)[6,6,4] -> ReLU -> MaxPool2D[3,3,4]
    //         -> AvgPool2D[1,1,4] -> Flatten[4,1] -> Linear(4->10)
    SpinalConfig().generateVerilog(
      Sequential(
        globalDataType = I8(),
        inputShape = Seq(8, 8, 1),
        layers = Seq(
          Conv2D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I32())),
          Requantize(shift = 4, targetType = I8()),
          ReLU(),
          MaxPool2D(poolSize = 2, stride = 2),
          AvgPool2D(poolSize = 2, stride = 2),
          Flatten(),
          Linear(inFeatures = 4, outFeatures = 10, customType = Some(I32())),
          Requantize(shift = 4, targetType = I8())
        ),
        axiConfig = axiConfig
      )
    )
  }

  test("Sequential with Tanh/Sigmoid activations compilation") {
    // Exercises the activation layers after a multi-channel AvgPool2D (lanes repack C->1)
    SpinalConfig().generateVerilog(
      Sequential(
        globalDataType = I8(),
        inputShape = Seq(8, 8, 1),
        layers = Seq(
          Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, customType = Some(I32())),
          Requantize(shift = 4, targetType = I8()),
          ReLU(),
          AvgPool2D(poolSize = 2, stride = 2),
          Tanh(),
          Sigmoid(),
          Flatten()
        ),
        axiConfig = axiConfig
      )
    )
  }

  test("Sequential with Cast to BF16 and float Softmax compilation") {
    // Exercises mid-network dtype change (SInt -> FloatML) feeding a float-domain head:
    // [6,6,1] -> Conv2D(K3) [4,4,1] -> Cast BF16 -> Flatten [1,16] -> Softmax over 16 logits
    SpinalConfig().generateVerilog(
      Sequential(
        globalDataType = I8(),
        inputShape = Seq(6, 6, 1),
        layers = Seq(
          Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3, customType = Some(I32())),
          Requantize(shift = 4, targetType = I8()),
          Cast(targetType = BF16()),
          Flatten(),
          Softmax()
        ),
        axiConfig = axiConfig
      )
    )
  }
}
