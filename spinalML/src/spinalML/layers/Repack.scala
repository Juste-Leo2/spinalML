package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

case class RepackLayer[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanesIn: Int,
  lanesOut: Int
) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanesIn))
    val y = master(Tensor(dataType, shape, lanesOut))
  }
  
  io.y <> repack(io.x, lanesOut)
}

object RepackLayer {
  def apply[T <: Data](x: Tensor[T], lanesOut: Int): Tensor[T] = {
    val comp = new RepackLayer(x.dataType, x.shape, x.lanes, lanesOut)
    comp.io.x <> x
    comp.io.y
  }
}
