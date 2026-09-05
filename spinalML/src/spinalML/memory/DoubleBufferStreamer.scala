// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._

/**
 * DoubleBufferStreamer acts as a read controller for the StreamDoubleBuffer.
 * It waits for a tile to be ready, then sequentially reads all elements and
 * pushes them out as a standard Stream. Once the entire tile is consumed,
 * it signals `nextTile` to swap the double buffer banks.
 */
case class DoubleBufferStreamer[T <: Data](dataType: HardType[T], depth: Int, lanes: Int) extends Component {
  val memSize = depth / lanes
  
  val io = new Bundle {
    // Interface to StreamDoubleBuffer
    val readAddr  = out UInt(log2Up(memSize) bits)
    val readData  = in Vec(dataType, lanes)
    val nextTile  = out Bool()
    val tileReady = in Bool()
    
    // Output Stream Interface
    val streamOut = master(Stream(Vec(dataType, lanes)))

    // Command-boundary re-arm pulse: resets the read FSM and flushes the
    // delivery FIFO so a back-to-back refetch can never keep streaming a
    // stale tile mid-flight (StreamDoubleBuffer reArm flips the banks
    // underneath an in-flight streamer otherwise).
    val reArm     = in Bool() default(False)
  }
  
  // State Machine for reading
  val readCounter = Counter(memSize)
  val isReading = RegInit(False)
  
  io.nextTile := False
  
  when(io.tileReady && !isReading) {
    isReading := True
  }
  
  // To handle the 1-cycle read latency from StreamDoubleBuffer cleanly with Stream backpressure,
  // we issue read requests and push the responses into a small FIFO.
  // The request stream drives the read addresses.
  val reqStream = Stream(UInt(log2Up(memSize) bits))
  reqStream.valid := isReading
  reqStream.payload := readCounter.value
  
  io.readAddr := reqStream.payload
  
  when(reqStream.fire) {
    readCounter.increment()
    when(readCounter.willOverflowIfInc) {
      isReading := False
      io.nextTile := True // Signal the double buffer to flip banks
    }
  }
  
  // Explicit FIFO to perfectly handle the 1-cycle BRAM read latency and downstream backpressure.
  val fifo = new StreamFifo(Vec(dataType, lanes), 16)
  io.streamOut << fifo.io.pop
  
  // We can only issue a read request if there is enough space in the FIFO for both
  // the data we are requesting now, AND any data that might already be in flight (1 cycle delay).
  // Thus, we require availability > 1 (room for at least 2 elements).
  reqStream.ready := fifo.io.availability > 1
  
  // The data arrives from the BRAM exactly 1 cycle after a successful request (fire).
  // Because we checked availability > 1, the FIFO is GUARANTEED to be ready to accept this push.
  val delayedValid = RegNext(reqStream.fire) init(False)
  fifo.io.push.valid := delayedValid
  fifo.io.push.payload := io.readData

  fifo.io.flush := False

  when(io.reArm) {
    isReading := False
    readCounter.clear()
    fifo.io.flush := True
    delayedValid := False
  }
}
