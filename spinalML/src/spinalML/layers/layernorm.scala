package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{rsqrt, scale_add}

case class LayerNorm1D[T <: Data](dataType: HardType[T], channels: Int, seqLen: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(channels, seqLen), lanes = channels))
    val gamma = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val beta = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(channels, seqLen), lanes = channels))
  }
  
  // Registers for static parameters
  val gammaReg = Reg(Vec(dataType, channels))
  val betaReg = Reg(Vec(dataType, channels))
  
  val state = RegInit(U"00")
  io.gamma.stream.ready := state === 0
  io.beta.stream.ready := state === 1
  
  when(io.gamma.stream.fire) {
    gammaReg := io.gamma.stream.payload
    state := 1
  }
  when(io.beta.stream.fire) {
    betaReg := io.beta.stream.payload
    state := 2
  }
  
  val runMode = state === 2
  
  // =========================================================================
  // PIPELINE ARCHITECTURE SKELETON
  // In a real implementation, calculating Mean and Variance across 1024 channels
  // requires a pipelined Adder Tree to meet timing. 
  // We use a passthrough placeholder here to establish the architectural flow.
  // =========================================================================
  
  val pipelinedX = io.x.stream.m2sPipe()
  
  // 1. Calculate Mean (Placeholder)
  val mean = pipelinedX.payload(0) // Dummy
  
  // 2. Calculate Variance (Placeholder)
  val variance = pipelinedX.payload(0) // Dummy
  
  // 3. Rsqrt using our LUT infrastructure
  // We instantiate RsqrtOp manually for the single variance value
  val rsqrtComp = spinalML.ops.RsqrtOp(dataType, Seq(1), lanes = 1)
  
  // Create a stream for the variance
  val varStream = Stream(Vec(dataType, 1))
  varStream.valid := pipelinedX.valid
  varStream.payload(0) := variance
  rsqrtComp.io.a.stream << varStream
  
  val invStdDev = rsqrtComp.io.c.stream
  pipelinedX.ready := invStdDev.ready
  
  // 4. Final Normalization & ScaleAdd
  // y = (x - mean) * invStdDev * gamma + beta
  val outPayload = Vec(dataType, channels)
  for (i <- 0 until channels) {
    // Dummy math for skeleton: y_i = x_i + gamma_i
    val px = pipelinedX.payload(i)
    val pa = gammaReg(i)
    
    (px, pa) match {
      case (vx: SInt, va: SInt) => outPayload(i).assignFrom((vx + va).resized.asInstanceOf[T])
      case (vx: UInt, va: UInt) => outPayload(i).assignFrom((vx + va).resized.asInstanceOf[T])
      case (vx: FloatML, va: FloatML) => outPayload(i).assignFrom(vx.asInstanceOf[T]) // Dummy
      case _ => throw new Exception("Unsupported data type")
    }
  }
  
  val outStream = Stream(Vec(dataType, channels))
  outStream.valid := invStdDev.valid
  outStream.payload := outPayload
  invStdDev.ready := outStream.ready
  
  io.y.stream << outStream.m2sPipe()
}
