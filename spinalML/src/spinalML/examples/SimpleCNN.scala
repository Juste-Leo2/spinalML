package spinalML.examples

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._
import spinalML.layers._
import spinalML.activations._
import spinalML.poolings._

/**
 * A functional 1D Convolutional Neural Network (CNN) in SpinalML.
 * 
 * Architecture:
 * Input (Shape: [16, 1], Lanes: 1)
 *  -> Conv1D (Kernel: 3, 1 channel) -> Output Shape: [14, 1], Lanes: 1
 *  -> BatchNorm1D
 *  -> ReLU
 *  -> MaxPool1D (Pool: 2) -> Output Shape: [7, 1], Lanes: 1
 *  -> Repack (Gearbox to 7 lanes) -> Lanes: 7
 *  -> Flatten -> Output Shape: [7, 1], Lanes: 7
 *  -> Linear (Out: 2) -> Output Shape: [2, 1], Lanes: 1
 *  -> Softmax -> Output Shape: [2, 1], Lanes: 2 (repack)
 */
case class SimpleCNN(dataType: HardType[Data]) extends Component {
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(16, 1), lanes = 1))
    val y = master(Tensor(dataType, Seq(2, 1), lanes = 2))
  }
  
  // --------------------------------------------------------
  // 1. Conv1D Layer (Single Channel Convolution)
  // --------------------------------------------------------
  val convW = Tensor(dataType, Seq(3, 1), lanes = 3)
  convW.stream.valid := True
  convW.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  
  val convB = Tensor(dataType, Seq(1, 1), lanes = 1)
  convB.stream.valid := True
  convB.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  
  val conv1dOut = Conv1D(io.x, convW, convB) // Output [14, 1], lanes 1
  
  // --------------------------------------------------------
  // 2. BatchNorm1D + ReLU
  // --------------------------------------------------------
  val bnGamma = Tensor(dataType, Seq(1), lanes = 1)
  val bnBeta = Tensor(dataType, Seq(1), lanes = 1)
  bnGamma.stream.valid := True
  bnBeta.stream.valid := True
  bnGamma.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  bnBeta.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  
  val bnOut = batchnorm(conv1dOut, bnGamma, bnBeta)
  
  val reluOut = relu(bnOut)
  
  // --------------------------------------------------------
  // 3. MaxPool1D
  // --------------------------------------------------------
  val poolIn = reshape(reluOut, Seq(14, 1))
  val poolOut = maxpool1d(poolIn, poolSize = 2, stride = 2)
  
  // --------------------------------------------------------
  // 4. Repack (Gearbox) to feed Linear
  // --------------------------------------------------------
  val poolReshaped = reshape(poolOut, Seq(1, 7))
  val repacked = repack(poolReshaped, newLanes = 7)
  
  // --------------------------------------------------------
  // 5. Linear (Fully Connected)
  // --------------------------------------------------------
  val linW1 = Tensor(dataType, Seq(7, 1), lanes = 7)
  linW1.stream.valid := True
  linW1.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  val linB1 = Tensor(dataType, Seq(1, 1), lanes = 1)
  linB1.stream.valid := True
  linB1.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  val forkStream = StreamFork(repacked.stream, 2)
  
  val repacked1 = Tensor(dataType, repacked.shape, repacked.lanes)
  repacked1.stream << forkStream(0)
  
  val repacked2 = Tensor(dataType, repacked.shape, repacked.lanes)
  repacked2.stream << forkStream(1)
  
  val outClass1 = Linear(repacked1, linW1, linB1)
  
  val linW2 = Tensor(dataType, Seq(7, 1), lanes = 7)
  linW2.stream.valid := True
  linW2.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  val linB2 = Tensor(dataType, Seq(1, 1), lanes = 1)
  linB2.stream.valid := True
  linB2.stream.payload.foreach(_.assignFromBits(B(0, dataType.getBitsWidth bits)))
  val outClass2 = Linear(repacked2, linW2, linB2)
  
  // Concat the 2 classes: outClass1 [1,1], outClass2 [1,1] -> [1,2] with lanes=2
  val concatOut = concatenate(outClass1, outClass2, axis = 1)
  val concatReshaped = reshape(concatOut, Seq(2, 1))
  
  // --------------------------------------------------------
  // 6. Softmax
  // --------------------------------------------------------
  val softmaxComp = Softmax1D(dataType, channels = 2, seqLen = 1)
  softmaxComp.io.x <> concatReshaped
  
  io.y <> softmaxComp.io.y
}
