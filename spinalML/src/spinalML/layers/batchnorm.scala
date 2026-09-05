// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

case class BatchNorm1D[T <: Data](dataType: HardType[T], channels: Int, seqLen: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
    val gamma = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val beta = slave(Tensor(dataType, Seq(channels), lanes = channels))
    val y = master(Tensor(dataType, Seq(seqLen, channels), lanes = channels))
  }
  
  // Registers to hold the static parameters (Scale and Shift)
  val gammaReg = Reg(Vec(dataType, channels))
  val betaReg = Reg(Vec(dataType, channels))
  
  val state = RegInit(U"00") // 0: Wait Gamma, 1: Wait Beta, 2: Run
  
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
  
  val outPayload = Vec(dataType, channels)
  for (i <- 0 until channels) {
    val px = io.x.stream.payload(i)
    val pa = gammaReg(i)
    val pb = betaReg(i)
    
    (px, pa, pb) match {
      case (vx: SInt, va: SInt, vb: SInt) =>
        outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
      case (vx: UInt, va: UInt, vb: UInt) =>
        outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
      case (vx: FloatML, va: FloatML, vb: FloatML) =>
        val mulRes = spinalML.utils.Float.mul(vx, va)
        val addRes = spinalML.utils.Float.add(mulRes, vb)
        outPayload(i).assignFrom(addRes.asInstanceOf[T])
      case _ => throw new Exception("Unsupported data type")
    }
  }
  
  val computeStream = Stream(Vec(dataType, channels))
  computeStream.valid := io.x.stream.valid && runMode
  computeStream.payload := outPayload
  io.x.stream.ready := computeStream.ready && runMode
  
  // Adds 1 pipeline stage. Vivado will map the MAC + Reg directly into DSP48.
  io.y.stream << computeStream.m2sPipe()
}

object batchnorm {
  def apply[T <: Data](x: Tensor[T], gamma: Tensor[T], beta: Tensor[T]): Tensor[T] = {
    val seqLen = x.shape(0)
    val channels = if (x.shape.length > 1) x.shape(1) else 1
    val comp = BatchNorm1D(x.dataType, channels, seqLen)
    comp.io.x <> x
    comp.io.gamma <> gamma
    comp.io.beta <> beta
    comp.io.y
  }
}
