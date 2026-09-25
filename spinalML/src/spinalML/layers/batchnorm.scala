// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.{RoundingConfig, RoundingMode}
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{RequantizeOp, repack}

case class BatchNorm1D[T <: Data](
  dataType: HardType[T],
  channels: Int,
  seqLen: Int,
  shift: Int = 0,
  rounding: RoundingMode = RoundingConfig.current
) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
    val gamma = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val beta = slave(Tensor(dataType, Seq(channels), lanes = channels))
    // Command-boundary re-arm: restart the gamma/beta load sequence so a new
    // weight generation can be accepted (LAY-01).
    val reArm = in Bool()
    val y = master(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
  }
  
  val gammaReg = Reg(Vec(dataType, channels))
  val betaReg = Reg(Vec(dataType, channels))
  
  val state = RegInit(U"00") // 0: Wait Gamma, 1: Wait Beta, 2: Run
  
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
  
  val outPayload = Vec(dataType, channels)
  val computeStream = Stream(Vec(dataType, channels))
  computeStream.valid := io.x.stream.valid && runMode

  // LAY-05: the SInt MAC reuses RequantizeOp (shift + rounding + saturation)
  // instead of the legacy 2's-complement wrap. UInt/Float paths are unchanged.
  dataType() match {
    case vIn: SInt =>
      val accWidth = vIn.getWidth * 2 + 1
      val rq = RequantizeOp(SInt(accWidth bits), dataType, Seq(1), channels, shift, rounding)
      for (i <- 0 until channels) {
        val vx = io.x.stream.payload(i).asInstanceOf[SInt]
        val va = gammaReg(i).asInstanceOf[SInt]
        val vb = betaReg(i).asInstanceOf[SInt]
        rq.io.a.stream.payload(i) := ((vx * va) + vb).resize(accWidth)
      }
      rq.io.a.stream.valid := computeStream.valid
      rq.io.c.stream.ready := computeStream.ready
      outPayload := rq.io.c.stream.payload
    case _: UInt =>
      for (i <- 0 until channels) {
        val vx = io.x.stream.payload(i).asInstanceOf[UInt]
        val va = gammaReg(i).asInstanceOf[UInt]
        val vb = betaReg(i).asInstanceOf[UInt]
        outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
      }
    case _: FloatML =>
      for (i <- 0 until channels) {
        val vx = io.x.stream.payload(i).asInstanceOf[FloatML]
        val va = gammaReg(i).asInstanceOf[FloatML]
        val vb = betaReg(i).asInstanceOf[FloatML]
        val mulRes = spinalML.utils.Float.mul(vx, va)
        val addRes = spinalML.utils.Float.add(mulRes, vb)
        outPayload(i).assignFrom(addRes.asInstanceOf[T])
      }
    case _ => throw new Exception("Unsupported data type")
  }

  computeStream.payload := outPayload
  io.x.stream.ready := computeStream.ready && runMode
  
  // Adds 1 pipeline stage. Vivado will map the MAC + Reg directly into DSP48.
  io.y.stream << computeStream.m2sPipe()
}

object batchnorm {
  def apply[T <: Data](
    x: Tensor[T],
    gamma: Tensor[T],
    beta: Tensor[T],
    outLanes: Int = -1,
    reArm: Option[Bool] = None,
    shift: Int = 0,
    rounding: RoundingMode = RoundingConfig.current
  ): Tensor[T] = {
    val seqLen = x.shape(0)
    val channels = if (x.shape.length > 1) x.shape(1) else 1
    val inX = if (x.lanes != channels) repack(x, channels) else x
    val inGamma = if (gamma.lanes != channels) repack(gamma, channels) else gamma
    val inBeta = if (beta.lanes != channels) repack(beta, channels) else beta

    val comp = BatchNorm1D(inX.dataType, channels, seqLen, shift, rounding)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.x <> inX
    comp.io.gamma <> inGamma
    comp.io.beta <> inBeta
    val rawY = comp.io.y
    val finalLanes = if (outLanes > 0) outLanes else channels
    if (rawY.lanes != finalLanes) repack(rawY, finalLanes) else rawY
  }
}
