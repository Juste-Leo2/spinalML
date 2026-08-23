package spinalML.memory

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

/**
 * TapBuffer: storage for a deferred consumer of a tensor stream.
 *
 * In a DAG topology, one producer can feed several consumers that pull at
 * different times. The immediate consumer reads the direct passthrough;
 * every deferred consumer owns a TapBuffer whose FIFO capacity equals the
 * full tensor size. Each input handshake mirrors one transaction into BOTH
 * outputs, and the exact-capacity FIFO guarantees a one-shot inference never
 * overflows the deferred path.
 */
case class TapBuffer[T <: Data](dataType: HardType[T], depth: Int, lanes: Int) extends Component {
  require(depth > 0, "TapBuffer depth must be positive")
  val entries = Math.max(1, depth / lanes)

  val io = new Bundle {
    val streamIn = slave(Stream(Vec(dataType, lanes)))
    val directOut = master(Stream(Vec(dataType, lanes)))
    val tapOut = master(Stream(Vec(dataType, lanes)))
  }

  // Deferred path: exact-capacity FIFO.
  val fifo = StreamFifo(Vec(dataType, lanes), entries)

  // Tee: one input handshake drives both outputs atomically.
  io.streamIn.ready := io.directOut.ready && fifo.io.push.ready
  io.directOut.valid := io.streamIn.valid
  io.directOut.payload := io.streamIn.payload
  fifo.io.push.valid := io.streamIn.valid
  fifo.io.push.payload := io.streamIn.payload

  io.tapOut << fifo.io.pop
}

object TapBuffer {
  /**
   * Splits a tensor stream for multiple consumers. The first returned tensor is
   * the direct passthrough; each subsequent tensor flows through its own
   * TapBuffer sized to hold the whole tensor.
   */
  def fork[T <: Data](src: Tensor[T], consumers: Int): Seq[Tensor[T]] = {
    require(consumers >= 1, "TapBuffer.fork requires at least one consumer")
    if (consumers == 1) return Seq(src)

    val elements = src.shape.product
    val entries = Math.max(1, (elements + src.lanes - 1) / src.lanes)
    val depthElements = entries * src.lanes

    var current = src.stream
    val outs = scala.collection.mutable.ArrayBuffer[Tensor[T]]()
    for (i <- 0 until consumers) {
      val out = Tensor(src.dataType, src.shape, src.lanes)
      if (i == consumers - 1) {
        out.stream << current
      } else {
        val tee = TapBuffer(src.dataType, depthElements, src.lanes)
        tee.io.streamIn << current
        out.stream << tee.io.directOut
        current = tee.io.tapOut
      }
      outs += out
    }
    outs.toSeq
  }
}
