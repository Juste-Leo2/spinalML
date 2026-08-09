package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class ReciprocalOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val bitWidth = dataType.getBitsWidth
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  // mathFn with divide-by-zero protection
  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))

  if (bitWidth <= 8) {
    val isFloat = dataType().isInstanceOf[FloatML]
    val (valFn, encodeFn) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (MathLUTs.floatValFn(f.expBits, f.mantBits), MathLUTs.floatEncodeFn(f.expBits, f.mantBits))
    } else {
      (MathLUTs.intValFn(bitWidth), MathLUTs.intEncodeFn(bitWidth))
    }
    
    val lutOp = UnaryLUTOp(dataType, shape, lanes, valFn, encodeFn, mathFn)
    lutOp.io.a <> io.a
    io.c <> lutOp.io.c
  } else {
    // PWL Approximation
    val indexBits = 8
    val numSegments = 1 << indexBits
    val isFloat = dataType().isInstanceOf[FloatML]
    
    val segmentIndexFn: T => UInt = (x: T) => {
      x.asBits(bitWidth - 1 downto bitWidth - indexBits).asUInt
    }
    
    val (expBits, mantBits) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (f.expBits, f.mantBits)
    } else (0, 0)
    
    val segmentFn = spinalML.utils.PWLLUTs.createSegmentFn(bitWidth, isFloat, expBits, mantBits, indexBits, mathFn)
    
    val pwlOp = spinalML.utils.UnaryPWLOp(dataType, shape, lanes, numSegments, segmentIndexFn, segmentFn)
    pwlOp.io.a <> io.a
    io.c <> pwlOp.io.c
  }
}

object reciprocal {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = ReciprocalOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
