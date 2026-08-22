package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{ExpOp, ReciprocalOp}

case class Softmax1D[T <: Data](dataType: HardType[T], channels: Int, seqLen: Int) extends Component {
  require(isPow2(channels), "Channels must be a power of 2 for AdderTree")
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
  }

  // Helper Math Functions
  def maxFn(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => Mux(vx > va, vx, va).asInstanceOf[T]
    case (vx: UInt, va: UInt) => Mux(vx > va, vx, va).asInstanceOf[T]
    case (vx: FloatML, va: FloatML) => spinalML.utils.Float.max(vx, va).asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  def add(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => (vx + va).resized.asInstanceOf[T]
    case (vx: UInt, va: UInt) => (vx + va).resized.asInstanceOf[T]
    case (vx: FloatML, va: FloatML) => spinalML.utils.Float.add(vx, va).asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  def sub(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => 
       val diff = vx -^ va // expands by 1 bit
       val minVal = -(1 << (vx.getBitsWidth - 1))
       val clamped = Mux(diff < minVal, S(minVal, vx.getBitsWidth bits), diff.resized)
       clamped.asInstanceOf[T]
    case (vx: UInt, va: UInt) => 
       val diff = vx.intoSInt -^ va.intoSInt
       val clamped = Mux(diff < 0, U(0, vx.getBitsWidth bits), diff.asUInt.resized)
       clamped.asInstanceOf[T]
    case (vx: FloatML, va: FloatML) => 
        val negB = FloatML(va.expBits, va.mantBits)
        negB.sign := !va.sign
        negB.exponent := va.exponent
        negB.mantissa := va.mantissa
        spinalML.utils.Float.add(vx, negB).asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  def mul(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => (vx * va).resized.asInstanceOf[T]
    case (vx: UInt, va: UInt) => (vx * va).resized.asInstanceOf[T]
    case (vx: FloatML, va: FloatML) => spinalML.utils.Float.mul(vx, va).asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  // Generic Pipelined Tree
  def buildPipelinedTree[C <: Data](
    inputStream: Stream[Vec[T]], 
    carryStream: Stream[C],
    treeFn: (T, T) => T
  ): (Stream[T], Stream[C]) = {
    case class StageBundle(len: Int) extends Bundle {
      val vals = Vec(dataType(), len)
      val carry = cloneOf(carryStream.payload)
    }
    var currentStream = Stream(StageBundle(inputStream.payload.length))
    
    currentStream.valid := inputStream.valid && carryStream.valid
    inputStream.ready := currentStream.ready && carryStream.valid
    carryStream.ready := currentStream.ready && inputStream.valid
    currentStream.payload.vals := inputStream.payload
    currentStream.payload.carry := carryStream.payload
    
    var currentLen = inputStream.payload.length
    
    while(currentLen > 1) {
      val nextLen = (currentLen + 1) / 2
      val nextStream = Stream(StageBundle(nextLen))
      
      val nextVals = Vec(dataType, nextLen)
      for (i <- 0 until currentLen / 2) {
         nextVals(i) := treeFn(currentStream.payload.vals(2*i), currentStream.payload.vals(2*i+1))
      }
      if (currentLen % 2 != 0) {
         nextVals(nextLen - 1) := currentStream.payload.vals(currentLen - 1)
      }
      
      nextStream.valid := currentStream.valid
      currentStream.ready := nextStream.ready
      nextStream.payload.vals := nextVals
      nextStream.payload.carry := currentStream.payload.carry
      
      currentStream = nextStream.m2sPipe()
      currentLen = nextLen
    }
    
    val (stream1, stream2) = StreamFork2(currentStream)
    val outVal = stream1.translateWith(stream1.payload.vals(0))
    val outCarry = stream2.translateWith(stream2.payload.carry)
    
    (outVal, outCarry)
  }

  val pipelinedX = io.x.stream.m2sPipe()
  val (pipelinedX1, pipelinedX2) = StreamFork2(pipelinedX)
  
  // 1. Max-Tree
  val (maxStream, carryXStream) = buildPipelinedTree(pipelinedX1, pipelinedX2, maxFn)
  
  // 2. Subtract Max: X' = X - max(X)
  case class ShiftBundle() extends Bundle {
    val shifted = Vec(dataType(), channels)
  }
  val shiftStream = Stream(ShiftBundle())
  
  shiftStream.valid := maxStream.valid && carryXStream.valid
  maxStream.ready := shiftStream.ready && carryXStream.valid
  carryXStream.ready := shiftStream.ready && maxStream.valid
  
  for(i <- 0 until channels) {
    shiftStream.payload.shifted(i) := sub(carryXStream.payload(i), maxStream.payload)
  }
  
  val shiftPipe = shiftStream.m2sPipe()
  
  // 3. ExpOp
  val expComp = spinalML.ops.ExpOp(dataType, Seq(channels), lanes = channels)
  val expInStream = Stream(Vec(dataType, channels))
  expInStream.valid := shiftPipe.valid
  expInStream.payload := shiftPipe.payload.shifted
  shiftPipe.ready := expInStream.ready
  
  expComp.io.a.stream << expInStream
  val expOutStream = expComp.io.c.stream
  
  // 4. Adder-Tree (sum of exp)
  val (expOut1, expOut2) = StreamFork2(expOutStream)
  val (sumExpStream, carryExpStream) = buildPipelinedTree(expOut1, expOut2, add)
  
  // 5. Reciprocal (1 / sum)
  val recipComp = spinalML.ops.ReciprocalOp(dataType, Seq(1), lanes = 1)
  
  val sumVecStream = Stream(Vec(dataType, 1))
  sumVecStream.valid := sumExpStream.valid
  sumVecStream.payload(0) := sumExpStream.payload
  sumExpStream.ready := sumVecStream.ready
  
  recipComp.io.a.stream << sumVecStream
  val invSumStream = recipComp.io.c.stream
  
  // 6. Final Multiply: Y = e^(X') * (1/sum)
  val outPayload = Vec(dataType, channels)

  val outStream = Stream(Vec(dataType, channels))
  val finalSyncValid = invSumStream.valid && carryExpStream.valid
  // Consume the join when the downstream pipe actually takes the beat
  // (its ready is !full || y.ready), NOT on raw io.y.stream.ready: otherwise
  // an empty pipe captures a pair without the join consuming it, and the
  // stored beat is re-emitted (duplicated) on the next y.ready pulse.
  invSumStream.ready := outStream.ready && carryExpStream.valid
  carryExpStream.ready := outStream.ready && invSumStream.valid

  for (i <- 0 until channels) {
    val eX = carryExpStream.payload(i)
    val invS = invSumStream.payload(0)
    outPayload(i) := mul(eX, invS)
  }

  outStream.valid := finalSyncValid
  outStream.payload := outPayload

  io.y.stream << outStream.m2sPipe()
}
