package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class RsqrtOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val bitWidth = dataType.getBitsWidth
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  if (bitWidth <= 8) {
    val isFloat = dataType().isInstanceOf[FloatML]
    val (valFn, encodeFn) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (MathLUTs.floatValFn(f.expBits, f.mantBits), MathLUTs.floatEncodeFn(f.expBits, f.mantBits))
    } else {
      (MathLUTs.intValFn(bitWidth), MathLUTs.intEncodeFn(bitWidth))
    }
    
    val lutOp = UnaryLUTOp(dataType, shape, lanes, valFn, encodeFn, (x: Double) => 1.0 / Math.sqrt(Math.abs(x) + 1e-5))
    lutOp.io.a <> io.a
    io.c <> lutOp.io.c
  } else {
    // Passthrough for larger types (BF16, I32) pending Piece-Wise Linear (PWL)
    io.c <> io.a
  }
}

object rsqrt {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = RsqrtOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
