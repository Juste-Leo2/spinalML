package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

// Hardware component for element-wise multiplication
case class MulOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  // SpinalHDL StreamJoin automatically handles the valid/ready handshake between a and b
  val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
  
  // Compute the combinatorial multiplication
  val payloadResult = Vec(dataType, lanes)
  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.b.stream.payload(i)) match {
      case (valA: SInt, valB: SInt) => payloadResult(i).assignFrom((valA * valB).resized.asInstanceOf[T])
      case (valA: UInt, valB: UInt) => payloadResult(i).assignFrom((valA * valB).resized.asInstanceOf[T])
      case _ => throw new Exception("Data type not supported for mul operation")
    }
  }
  
  // Pipeline the output to optimize max clock frequency (DSP blocks run faster if registered)
  io.c.stream << syncStream.translateWith(payloadResult).m2sPipe()
}

object mul {
  /**
   * Element-wise tensor multiplication.
   */
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(a.shape == b.shape, "Tensors must have the same shape")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    
    val mulComp = MulOp(a.dataType, a.shape, a.lanes)
    mulComp.io.a <> a
    mulComp.io.b <> b
    mulComp.io.c
  }
}
