// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

case class MulOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int, dspConfig: spinalML.dsp.DspConfig = spinalML.dsp.DspConfig.default) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
  
  val payloadResult = Vec(dataType, lanes)
  for (i <- 0 until lanes) {
    payloadResult(i) := spinalML.dsp.DspMul(
      a = io.a.stream.payload(i),
      b = io.b.stream.payload(i),
      enable = syncStream.fire,
      accType = dataType,
      latency = 0,
      dspConfig = dspConfig
    )
  }
  
  // Pipeline the output to optimize max clock frequency (DSP blocks run faster if registered)
  io.c.stream << syncStream.translateWith(payloadResult).m2sPipe()
}

object mul {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T], dspConfig: spinalML.dsp.DspConfig = spinalML.dsp.DspConfig.default): Tensor[T] = {
    require(a.shape == b.shape, "Tensors must have the same shape")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    
    val mulComp = MulOp(a.dataType, a.shape, a.lanes, dspConfig = dspConfig)
    mulComp.io.a <> a
    mulComp.io.b <> b
    mulComp.io.c
  }
}
