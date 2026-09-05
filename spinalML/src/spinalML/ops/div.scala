// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

case class DivOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  // 1. Synchronize inputs
  val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
  val (syncStreamForB, syncStreamForA) = StreamFork2(syncStream)

  // 2. Calculate Reciprocal of B
  val invBComp = ReciprocalOp(dataType, shape, lanes)
  invBComp.io.a.stream << syncStreamForB.translateWith(io.b.stream.payload)
  
  // 3. Pipeline A to match the 1-cycle latency of ReciprocalOp
  val pipelinedA = syncStreamForA.translateWith(io.a.stream.payload).m2sPipe()
  val invBStream = invBComp.io.c.stream
  
  // 4. Join streams at output (safety net, they should arrive together)
  val joined = StreamJoin.arg(pipelinedA, invBStream)
  
  val outValid = RegInit(False)
  when(io.c.stream.ready || !outValid) {
    outValid := joined.valid
  }

  val outPayload = Reg(Vec(dataType, lanes))
  when(joined.valid && (io.c.stream.ready || !outValid)) {
    for (i <- 0 until lanes) {
      val pa = pipelinedA.payload(i)
      val pInvB = invBStream.payload(i)
      
      (pa, pInvB) match {
        case (va: SInt, vInvB: SInt) =>
          outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
        case (va: UInt, vInvB: UInt) =>
          outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
        case (va: FloatML, vInvB: FloatML) =>
          outPayload(i).assignFrom(spinalML.utils.Float.mul(va, vInvB).asInstanceOf[T])
        case _ => throw new Exception("Unsupported data type for Div")
      }
    }
  }

  joined.ready := io.c.stream.ready || !outValid
  io.c.stream.valid := outValid
  io.c.stream.payload := outPayload
}

object div {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    val comp = DivOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.b <> b
    comp.io.c
  }
}
