// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{rsqrt, scale_add, repack}

case class LayerNorm1D[T <: Data](dataType: HardType[T], channels: Int, seqLen: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
    val gamma = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val beta = slave(Tensor(dataType, Seq(channels), lanes = channels))
    // Command-boundary re-arm: restart the gamma/beta load sequence so a new
    // weight generation can be accepted (LAY-01).
    val reArm = in Bool()
    val y = master(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
  }
  
  // Registers for static parameters
  val gammaReg = Reg(Vec(dataType, channels))
  val betaReg = Reg(Vec(dataType, channels))
  
  val state = RegInit(U"00")
  io.gamma.stream.ready := state === 0
  io.beta.stream.ready := state === 1

  // LAY-01: a command boundary restarts the load sequence so a new gamma/beta
  // generation is accepted. Placed before the fire assignments so a same-cycle
  // valid beat of the new generation still wins.
  when(io.reArm) {
    state := 0
  }
  when(io.gamma.stream.fire) {
    gammaReg := io.gamma.stream.payload
    state := 1
  }
  when(io.beta.stream.fire) {
    betaReg := io.beta.stream.payload
    state := 2
  }
  
  val runMode = state === 2
  
  require(isPow2(channels), "Channels must be a power of 2 for division by shift")
  
  def add(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => (vx + va).resized.asInstanceOf[T]
    case (vx: UInt, va: UInt) => (vx + va).resized.asInstanceOf[T]
    case (vx: FloatML, va: FloatML) => spinalML.utils.Float.add(vx, va).asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  def sub(a: T, b: T): T = (a, b) match {
    case (vx: SInt, va: SInt) => (vx - va).resized.asInstanceOf[T]
    case (vx: UInt, va: UInt) => (vx - va).resized.asInstanceOf[T]
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

  def divN(a: T, N: Int): T = a match {
    case (vx: SInt) => (vx / N).resized.asInstanceOf[T]
    case (vx: UInt) => (vx / N).resized.asInstanceOf[T]
    case (vx: FloatML) => 
        val res = FloatML(vx.expBits, vx.mantBits)
        res.sign := vx.sign
        res.mantissa := vx.mantissa
        val logN = spinal.core.log2Up(N)
        val expSInt = vx.exponent.intoSInt - logN
        when(vx.exponent === 0 || expSInt <= 0) {
            res.exponent := 0
            res.mantissa := 0
            res.sign := False
        } otherwise {
            res.exponent := expSInt.asUInt.resized
        }
        res.asInstanceOf[T]
    case _ => throw new Exception("Unsupported type")
  }

  // Generic Pipelined Adder Tree that carries an arbitrary payload `carry`
  def buildPipelinedTree[C <: Data](
    inputStream: Stream[Vec[T]], 
    carryStream: Stream[C],
    addFn: (T, T) => T
  ): (Stream[T], Stream[C]) = {
    case class StageBundle(len: Int) extends Bundle {
      val sums = Vec(dataType(), len)
      val carry = cloneOf(carryStream.payload)
    }
    var currentStream = Stream(StageBundle(inputStream.payload.length))
    
    currentStream.valid := inputStream.valid && carryStream.valid
    inputStream.ready := currentStream.ready && carryStream.valid
    carryStream.ready := currentStream.ready && inputStream.valid
    currentStream.payload.sums := inputStream.payload
    currentStream.payload.carry := carryStream.payload
    
    var currentLen = inputStream.payload.length
    
    while(currentLen > 1) {
      val nextLen = (currentLen + 1) / 2
      val nextStream = Stream(StageBundle(nextLen))
      
      val nextSums = Vec(dataType, nextLen)
      for (i <- 0 until currentLen / 2) {
         nextSums(i) := addFn(currentStream.payload.sums(2*i), currentStream.payload.sums(2*i+1))
      }
      if (currentLen % 2 != 0) {
         nextSums(nextLen - 1) := currentStream.payload.sums(currentLen - 1)
      }
      
      nextStream.valid := currentStream.valid
      currentStream.ready := nextStream.ready
      nextStream.payload.sums := nextSums
      nextStream.payload.carry := currentStream.payload.carry
      
      currentStream = nextStream.m2sPipe()
      currentLen = nextLen
    }
    
    val (fork1, fork2) = StreamFork2(currentStream)
    
    val outSum = fork1.translateWith(fork1.payload.sums(0))
    val outCarry = fork2.translateWith(fork2.payload.carry)
    
    (outSum, outCarry)
  }

  val pipelinedX = io.x.stream.m2sPipe()
  
  // Fork pipelinedX because buildPipelinedTree expects two separate streams
  val (pipelinedX1, pipelinedX2) = StreamFork2(pipelinedX)
  
  val (sumStream, carryXStream) = buildPipelinedTree(pipelinedX1, pipelinedX2, add)
  
  val muStream = sumStream.translateWith(divN(sumStream.payload, channels))
  
  case class DiffBundle() extends Bundle {
    val diffs = Vec(dataType(), channels)
    val origDiffs = Vec(dataType(), channels)
  }
  val diffStream = Stream(DiffBundle())
  
  diffStream.valid := muStream.valid && carryXStream.valid
  muStream.ready := diffStream.ready && carryXStream.valid
  carryXStream.ready := diffStream.ready && muStream.valid
  
  for(i <- 0 until channels) {
    val x = carryXStream.payload(i)
    val mu = muStream.payload
    val diff = sub(x, mu)
    diffStream.payload.diffs(i) := mul(diff, diff) // (x - mu)^2
    diffStream.payload.origDiffs(i) := diff // carry (x - mu)
  }
  
  val diffPipe = diffStream.m2sPipe()
  val (diffPipe1, diffPipe2) = StreamFork2(diffPipe)
  
  val diffsOnly = diffPipe1.translateWith(diffPipe1.payload.diffs)
  val origDiffsOnly = diffPipe2.translateWith(diffPipe2.payload.origDiffs)
  
  val (varSumStream, carryDiffsStream) = buildPipelinedTree(diffsOnly, origDiffsOnly, add)
  val varStream = varSumStream.translateWith(divN(varSumStream.payload, channels))
  
  val rsqrtComp = spinalML.ops.RsqrtOp(dataType, Seq(1), lanes = 1)
  
  val epsVal: T = (dataType() match {
    case f: FloatML => 
      // 1e-5 must stay representable, otherwise a zero variance (or a
      // variance whose diff^2 underflowed) reaches the rsqrt LUT at input 0 and
      // the guard saturates the inverse standard deviation. If the encoded
      // epsilon underflows to 0, fall back to the smallest positive normal
      // (exp=1, mant=0): 2^-6 in E4M3, 1.0 in E2M1. BF16 keeps 1e-5. The
      // fallback never applies to a format where 1e-5 is representable, so no
      // existing dtype changes beyond the underflowing ones (FTZ has no
      // subnormal escape hatch).
      val epsEncoded = spinalML.utils.MathLUTs.floatEncodeFn(f.expBits, f.mantBits)(1e-5)
      val epsLiteral: BigInt = if (epsEncoded != 0) epsEncoded else BigInt(1 << f.mantBits)
      val epsBits = B(epsLiteral, f.getBitsWidth bits)
      val epsFloat = FloatML(f.expBits, f.mantBits)
      epsFloat.assignFromBits(epsBits)
      epsFloat
    case s: SInt => S(0, s.getWidth bits)
    case u: UInt => U(0, u.getWidth bits)
    case _ => throw new Exception("Unsupported type")
  }).asInstanceOf[T]

  val varWithEpsStream = varStream.translateWith(add(varStream.payload, epsVal))

  val varVecStream = Stream(Vec(dataType, 1))
  varVecStream.valid := varWithEpsStream.valid
  varVecStream.payload(0) := varWithEpsStream.payload
  varWithEpsStream.ready := varVecStream.ready
  
  rsqrtComp.io.a.stream << varVecStream
  val invStdDevStream = rsqrtComp.io.c.stream
  
  // y_i = (x_i - mu) * invStdDev * gamma_i + beta_i
  val outPayload = Vec(dataType, channels)
  
  // Synchronize invStdDevStream and carryDiffsStream
  val finalSyncValid = invStdDevStream.valid && carryDiffsStream.valid
  invStdDevStream.ready := io.y.stream.ready && carryDiffsStream.valid
  carryDiffsStream.ready := io.y.stream.ready && invStdDevStream.valid
  
  for (i <- 0 until channels) {
    val diff = carryDiffsStream.payload(i)
    val invStdDev = invStdDevStream.payload(0)
    val gamma = gammaReg(i)
    val beta = betaReg(i)
    
    val norm = mul(diff, invStdDev)
    val scaled = mul(norm, gamma)
    outPayload(i) := add(scaled, beta)
  }
  
  val outStream = Stream(Vec(dataType, channels))
  outStream.valid := finalSyncValid
  outStream.payload := outPayload
  
  io.y.stream << outStream.m2sPipe()
}

object layernorm {
  def apply[T <: Data](x: Tensor[T], gamma: Tensor[T], beta: Tensor[T], outLanes: Int = -1, reArm: Option[Bool] = None): Tensor[T] = {
    val seqLen = x.shape(0)
    val channels = if (x.shape.length > 1) x.shape(1) else 1
    val inX = if (x.lanes != channels) repack(x, channels) else x
    val inGamma = if (gamma.lanes != channels) repack(gamma, channels) else gamma
    val inBeta = if (beta.lanes != channels) repack(beta, channels) else beta

    val comp = LayerNorm1D(inX.dataType, channels, seqLen)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.x <> inX
    comp.io.gamma <> inGamma
    comp.io.beta <> inBeta
    val rawY = comp.io.y
    val finalLanes = if (outLanes > 0) outLanes else channels
    if (rawY.lanes != finalLanes) repack(rawY, finalLanes) else rawY
  }
}
