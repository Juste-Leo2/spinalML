package spinalML.tensors

import spinal.core._
import spinal.lib._

/**
 * Generic Tensor representation in hardware using Streaming (Pipelined).
 * Data flows through the `stream` payload in chunks defined by `lanes`.
 */
case class Tensor[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Bundle with IMasterSlave {
  require(shape.nonEmpty, "Tensor shape cannot be empty")
  require(shape.forall(_ > 0), "All dimensions must be > 0")

  val totalElements = shape.product
  // require(totalElements % lanes == 0, s"Total elements ($totalElements) must be a multiple of lanes ($lanes)")

  val stream = Stream(Vec(dataType, lanes))

  override def asMaster(): Unit = master(stream)
}

object Tensor {
  // GGML inspired constructors
  
  def Tensor1D[T <: Data](dataType: HardType[T], ne0: Int, lanes: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0), lanes)
  }

  def Tensor2D[T <: Data](dataType: HardType[T], ne0: Int, ne1: Int, lanes: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0, ne1), lanes)
  }

  def Tensor3D[T <: Data](dataType: HardType[T], ne0: Int, ne1: Int, ne2: Int, lanes: Int): Tensor[T] = {
    Tensor(dataType, Seq(ne0, ne1, ne2), lanes)
  }
}
