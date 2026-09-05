// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

case class SubOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  // SpinalHDL StreamJoin automatically handles the valid/ready handshake between a and b
  val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
  
  val payloadResult = Vec(dataType, lanes)
  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.b.stream.payload(i)) match {
      case (valA: SInt, valB: SInt) => payloadResult(i).assignFrom((valA - valB).asInstanceOf[T])
      case (valA: UInt, valB: UInt) => payloadResult(i).assignFrom((valA - valB).asInstanceOf[T])
      case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => {
        val invertedB = spinalML.dtypes.FloatML(valB.expBits, valB.mantBits)
        invertedB.exponent := valB.exponent
        invertedB.mantissa := valB.mantissa
        invertedB.sign := !valB.sign
        payloadResult(i).assignFrom(spinalML.utils.Float.add(valA, invertedB).asInstanceOf[T])
      }
      case _ => throw new Exception("Type de donnée non supporté pour l'opération sub")
    }
  }
  
  io.c.stream << syncStream.translateWith(payloadResult).m2sPipe()
}

object sub {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(a.shape == b.shape, "Les Tensors doivent avoir la même forme (shape)")
    require(a.lanes == b.lanes, "Les Tensors d'entrée doivent avoir la même largeur (lanes)")
    
    val subComp = SubOp(a.dataType, a.shape, a.lanes)
    subComp.io.a <> a
    subComp.io.b <> b
    subComp.io.c
  }
}
