// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{ExpOp, ReciprocalOp}

case class Softmax1D[T <: Data](dataType: HardType[T], channels: Int, seqLen: Int) extends Component {
  require(channels > 0, "channels must be >= 1")
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
  def buildPipelinedTree[C <: Data, N <: Data](
    inputStream: Stream[Vec[T]], 
    carryStream: Stream[C],
    nodeType: HardType[N],
    toNode: T => N,
    treeFn: (N, N) => N
  ): (Stream[N], Stream[C]) = {
    case class StageBundle(len: Int) extends Bundle {
      val vals = Vec(nodeType(), len)
      val carry = cloneOf(carryStream.payload)
    }
    var currentStream = Stream(StageBundle(inputStream.payload.length))
    
    currentStream.valid := inputStream.valid && carryStream.valid
    inputStream.ready := currentStream.ready && carryStream.valid
    carryStream.ready := currentStream.ready && inputStream.valid
    for (i <- 0 until inputStream.payload.length) {
      currentStream.payload.vals(i) := toNode(inputStream.payload(i))
    }
    currentStream.payload.carry := carryStream.payload
    
    var currentLen = inputStream.payload.length
    
    while(currentLen > 1) {
      val nextLen = (currentLen + 1) / 2
      val nextStream = Stream(StageBundle(nextLen))
      
      val nextVals = Vec(nodeType, nextLen)
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
  val (maxStream, carryXStream) = buildPipelinedTree(pipelinedX1, pipelinedX2, dataType, (v: T) => v, maxFn)
  
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
  
  // 4. Adder-Tree (sum of exp) — accumulated in a wide format to avoid the
  // rounding of every pair in the output dtype (FP32 for floats, 
  // dataWidth + log2(channels) for ints), then rounded/saturated ONCE at the
  // reciprocal input. Bit-exact for the golden model: same tree order.
  val isFloatAcc = dataType().isInstanceOf[FloatML]
  
  val (expOut1, expOut2) = StreamFork2(expOutStream)
  val (sumVecStream, carryExpStream) = if (isFloatAcc && dataType().asInstanceOf[FloatML].mantBits > 4) {
    // Wide floats: accumulate in an FP32 tree (single round to the dtype at
    // the reciprocal input). Bit-exact with the Python golden: same tree order.
    val fml = dataType().asInstanceOf[FloatML]
    val accNode = HardType(FloatML(8, 23))
    val to32: T => FloatML = (v: T) => spinalML.utils.Float.widen(v.asInstanceOf[FloatML], 8, 23)
    val add32: (FloatML, FloatML) => FloatML = (a, b) => spinalML.utils.Float.add(a, b)
    val (sumExpStream, carryExpStream) = buildPipelinedTree(expOut1, expOut2, accNode, to32, add32)
    
    val s = Stream(Vec(dataType, 1))
    s.valid := sumExpStream.valid
    s.payload(0).assignFromBits(
      spinalML.utils.Float.roundTo(sumExpStream.payload, fml.expBits, fml.mantBits).asBits
    )
    sumExpStream.ready := s.ready
    (s, carryExpStream)
  } else if (isFloatAcc) {
    // Narrow floats (mantBits <= 4, e.g. FP8): exact integer block-float
    // accumulator. Every exp value is aligned on a FIXED reference exponent
    // (exp field 1 = LSB): the small exponent span guarantees no truncation,
    // so the sum is exact — then normalized ONCE back to the dtype (RNE).
    // Much cheaper than an FP32 tree and strictly more accurate.
    val fmlN = dataType().asInstanceOf[FloatML]
    val expBits = fmlN.expBits
    val mantBits = fmlN.mantBits
    val spanBits = (1 << expBits) - 2 // max shift = max exp field - 1
    val mantIntBits = mantBits + 1
    val accWidth = mantIntBits + spanBits + log2Up(channels)
    val accNode = HardType(UInt(accWidth bits))

    val toAcc: T => UInt = (v: T) => {
      val x = v.asInstanceOf[FloatML]
      val shift = Mux(x.exponent === 0, U(0, expBits bits), (x.exponent - U(1, expBits bits)).resize(expBits))
      val mantC = (U((1 << mantBits), mantIntBits bits) | x.mantissa.resize(mantIntBits)).resize(mantIntBits)
      val mant = Mux(x.exponent === 0, U(0, mantIntBits bits), mantC)
      (mant << shift).resized
    }
    val addAcc: (UInt, UInt) => UInt = (a, b) => (a + b).resized
    val (sumExpStream, carryExpStream) = buildPipelinedTree(expOut1, expOut2, accNode, toAcc, addAcc)

    // Normalize the exact fixed-point sum back to the dtype (RNE, clamp).
    val sumInt = sumExpStream.payload
    val msbBits = log2Up(accWidth + 1)
    var msbAcc = U(0, msbBits bits)
    for (i <- 0 until accWidth) {
      msbAcc = Mux(sumInt(i), U(i, msbBits bits), msbAcc)
    }
    val msb = Mux(!sumInt.orR, U(0, msbBits bits), msbAcc)

    val mantBitsU = U(mantBits, msbBits bits)
    val low = msb - mantBitsU // LSB index of the mantissa (wraps if tiny)
    val pGtM = msb > mantBitsU
    val lowC = Mux(pGtM, low, U(0, msbBits bits))
    val lowShL = mantBitsU - msb // left shift when p <= mantBits
    val gIdx = low - U(1) // reliable only when gValid; guarded below
    val maskBelowG = ((U(1, accWidth bits) << gIdx) - U(1, accWidth bits)).resize(accWidth)
    val guardB = Mux(pGtM, (sumInt >> gIdx)(0), False)
    val stickyB = Mux(pGtM, (sumInt & maskBelowG).orR, False)

    val maskBelowM = ((U(1, accWidth bits) << msb) - U(1, accWidth bits)).resize(accWidth)
    val raw = sumInt & maskBelowM
    val mantValRaw = Mux(pGtM, (raw >> lowC).resize(mantBits + 1), (raw << lowShL).resize(mantBits + 1))
    val mantVal = mantValRaw(mantBits - 1 downto 0)
    val roundUp = guardB & (mantVal(0) | stickyB)
    val mantRw = mantVal +^ roundUp.asUInt
    val mantOv = mantRw(mantBits)
    val mantF = Mux(mantOv, U(0, mantBits bits), mantRw(mantBits - 1 downto 0))

    val expRaw = msb - mantBitsU + U(1, msbBits bits) // int = v * 2^(bias+mantBits-1)
    val expF = Mux(mantOv, expRaw + U(1), expRaw)
    val expMax = (1 << expBits) - 1
    val expOv = expF > U(expMax, msbBits bits)

    val s = Stream(Vec(dataType, 1))
    val zeroSum = !sumInt.orR
    val sumF = s.payload(0).asInstanceOf[FloatML]
    sumF.sign := False
    sumF.exponent := Mux(expOv, U(expMax, expBits bits), Mux(zeroSum, U(0, expBits bits), expF.resize(expBits)))
    sumF.mantissa := Mux(expOv, U(0, mantBits bits), mantF)
    s.valid := sumExpStream.valid
    sumExpStream.ready := s.ready
    (s, carryExpStream)
  } else {
    // Int accumulation: exact in (dataWidth + log2Up(channels)), saturating
    // once to the output dtype at the reciprocal input (exp values are >= 0).
    val accWidth = dataType.getBitsWidth + log2Up(channels)
    val accNode = HardType(UInt(accWidth bits))
    val toAcc: T => UInt = (v: T) => v.asBits.asUInt.resize(accWidth)
    val addAcc: (UInt, UInt) => UInt = (a, b) => (a + b).resized
    val (sumExpStream, carryExpStream) = buildPipelinedTree(expOut1, expOut2, accNode, toAcc, addAcc)
    
    val s = Stream(Vec(dataType, 1))
    val maxRep = (1 << (dataType.getBitsWidth - 1)) - 1
    val sat = Mux(sumExpStream.payload > U(maxRep, accWidth bits), U(maxRep, accWidth bits), sumExpStream.payload)
    s.valid := sumExpStream.valid
    s.payload(0).assignFromBits(sat.resize(dataType.getBitsWidth).asBits)
    sumExpStream.ready := s.ready
    (s, carryExpStream)
  }
  
  // 5. Reciprocal (1 / sum)
  val recipComp = spinalML.ops.ReciprocalOp(dataType, Seq(1), lanes = 1)
  
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
