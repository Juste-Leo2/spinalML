package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

case class AbsOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  io.c.stream << io.a.stream.map { payload =>
    val outPayload = Vec(dataType, lanes)
    for (i <- 0 until lanes) {
      payload(i) match {
        case valA: SInt => 
          outPayload(i).assignFrom(Mux(valA < 0, -valA, valA).asInstanceOf[T])
        case valA: UInt =>
          outPayload(i).assignFrom(valA.asInstanceOf[T])
        case valA: spinalML.dtypes.FloatML =>
          val outF = spinalML.dtypes.FloatML(valA.expBits, valA.mantBits)
          outF.sign := False
          outF.exponent := valA.exponent
          outF.mantissa := valA.mantissa
          outPayload(i).assignFrom(outF.asInstanceOf[T])
        case _ => 
          throw new Exception("Unsupported data type for Abs")
      }
    }
    outPayload
  }.m2sPipe()
}

object abs {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = AbsOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
