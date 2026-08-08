package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

case class ReLUOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanes))
    val y = master(Tensor(dataType, shape, lanes))
  }
  
  // Pass through the stream control signals
  io.y.stream.arbitrationFrom(io.x.stream)
  
  for (i <- 0 until lanes) {
    io.x.stream.payload(i) match {
      case valX: SInt => 
        val zero = SInt(valX.getWidth bits)
        zero := 0
        io.y.stream.payload(i).assignFrom(Mux(valX < 0, zero, valX).asInstanceOf[T])
      case valX: UInt => 
        io.y.stream.payload(i).assignFrom(valX.asInstanceOf[T])
      case valX: spinalML.dtypes.FloatML =>
        io.y.stream.payload(i).assignFrom(Mux(valX.sign, spinalML.utils.Float.zero(valX.expBits, valX.mantBits), valX).asInstanceOf[T])
      case _ => 
        throw new Exception("Data type not supported for ReLU operation")
    }
  }
}

object relu {
  def apply[T <: Data](x: Tensor[T]): Tensor[T] = {
    val reluComp = ReLUOp(x.dataType, x.shape, x.lanes)
    reluComp.io.x <> x
    reluComp.io.y
  }
}
