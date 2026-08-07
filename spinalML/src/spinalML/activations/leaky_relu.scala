package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

case class LeakyReLUOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int, shift: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanes))
    val y = master(Tensor(dataType, shape, lanes))
  }
  
  io.y.stream.arbitrationFrom(io.x.stream)
  
  for (i <- 0 until lanes) {
    io.x.stream.payload(i) match {
      case valX: SInt => 
        // Arithmetic right shift for negative slope
        io.y.stream.payload(i).assignFrom(Mux(valX < 0, valX >> shift, valX).asInstanceOf[T])
      case valX: UInt => 
        io.y.stream.payload(i).assignFrom(valX.asInstanceOf[T])
      case _ => 
        throw new Exception("Data type not supported for LeakyReLU operation")
    }
  }
}

object leaky_relu {
  def apply[T <: Data](x: Tensor[T], shift: Int): Tensor[T] = {
    val comp = LeakyReLUOp(x.dataType, x.shape, x.lanes, shift)
    comp.io.x <> x
    comp.io.y
  }
}
