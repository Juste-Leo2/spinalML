package spinalML.ops

import spinal.core._
import spinalML.tensors.Tensor

case class AddOp[T <: Data](dataType: HardType[T], shape: Seq[Int]) extends Component {
  val io = new Bundle {
    val a = in(Tensor(dataType, shape))
    val b = in(Tensor(dataType, shape))
    val c = out(Tensor(dataType, shape))
  }
  
  for (i <- 0 until io.a.totalElements) {
    (io.a.data(i), io.b.data(i)) match {
      case (valA: SInt, valB: SInt) => io.c.data(i).assignFrom((valA + valB).asInstanceOf[T])
      case (valA: UInt, valB: UInt) => io.c.data(i).assignFrom((valA + valB).asInstanceOf[T])
      case _ => throw new Exception("Type de donnée non supporté pour l'opération add")
    }
  }
}

object add {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(a.shape == b.shape, "Les Tensors doivent avoir la même forme (shape)")
    val addComp = AddOp(a.dataType, a.shape)
    addComp.io.a := a
    addComp.io.b := b
    addComp.io.c
  }
}
