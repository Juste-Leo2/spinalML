package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

case class CastOp[TIn <: Data, TOut <: Data](
  dataTypeIn: HardType[TIn],
  dataTypeOut: HardType[TOut],
  shape: Seq[Int],
  lanes: Int
) extends Component {

  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, shape, lanes))
    val c = master(Tensor(dataTypeOut, shape, lanes))
  }

  // Pass through the stream control signals
  io.c.stream.arbitrationFrom(io.a.stream)

  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.c.stream.payload(i)) match {
      case (valIn: SInt, valOut: FloatML) =>
        io.c.stream.payload(i).assignFrom(
          spinalML.utils.Float.fromSInt(valIn, valOut.expBits, valOut.mantBits).asInstanceOf[TOut]
        )
      // More cases can be added here if needed in the future (e.g., UInt -> Float, Float -> SInt, etc.)
      case _ =>
        throw new Exception("Type de cast non supporté (seul SInt -> FloatML est géré pour le moment)")
    }
  }
}

object cast {
  def apply[TIn <: Data, TOut <: Data](a: Tensor[TIn], dataTypeOut: HardType[TOut]): Tensor[TOut] = {
    val castComp = CastOp(a.dataType, dataTypeOut, a.shape, a.lanes)
    castComp.io.a <> a
    castComp.io.c
  }
}
