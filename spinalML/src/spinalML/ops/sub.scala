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
  
  // Output stream valid when inputs are synchronized
  io.c.stream.arbitrationFrom(syncStream)
  
  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.b.stream.payload(i)) match {
      case (valA: SInt, valB: SInt) => io.c.stream.payload(i).assignFrom((valA - valB).asInstanceOf[T])
      case (valA: UInt, valB: UInt) => io.c.stream.payload(i).assignFrom((valA - valB).asInstanceOf[T])
      case _ => throw new Exception("Type de donnée non supporté pour l'opération sub")
    }
  }
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
