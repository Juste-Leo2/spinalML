package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

case class ScaleAddOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanes))
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  val joined = StreamJoin.arg(io.x.stream, io.a.stream, io.b.stream)
  
  val outValid = RegInit(False)
  when(io.c.stream.ready || !outValid) {
    outValid := joined.valid
  }

  val outPayload = Reg(Vec(dataType, lanes))
  when(joined.valid && (io.c.stream.ready || !outValid)) {
    for (i <- 0 until lanes) {
      val px = io.x.stream.payload(i)
      val pa = io.a.stream.payload(i)
      val pb = io.b.stream.payload(i)
      
      (px, pa, pb) match {
        case (vx: SInt, va: SInt, vb: SInt) =>
          outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
        case (vx: UInt, va: UInt, vb: UInt) =>
          outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
        case (vx: FloatML, va: FloatML, vb: FloatML) =>
          val mulRes = spinalML.utils.Float.mul(vx, va)
          val addRes = spinalML.utils.Float.add(mulRes, vb)
          outPayload(i).assignFrom(addRes.asInstanceOf[T])
        case _ => throw new Exception("Unsupported data type for ScaleAdd")
      }
    }
  }

  joined.ready := io.c.stream.ready || !outValid
  io.c.stream.valid := outValid
  io.c.stream.payload := outPayload
}

object scale_add {
  def apply[T <: Data](x: Tensor[T], a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    val comp = ScaleAddOp(x.dataType, x.shape, x.lanes)
    comp.io.x <> x
    comp.io.a <> a
    comp.io.b <> b
    comp.io.c
  }
}
