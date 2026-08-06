package spinalML.memory

import spinal.core._
import spinal.lib._

/**
 * StreamDoubleBuffer: A generic double-buffering component.
 * It reads from a Stream into Ping/Pong memory banks (BRAM).
 * Exposes a random-access read interface for the computation unit.
 * Handshake interface (nextTile, tileReady) controls the Ping/Pong switching.
 */
case class StreamDoubleBuffer[T <: Data](dataType: HardType[T], depth: Int, lanes: Int) extends Component {
  val io = new Bundle {
    // Input Stream
    val streamIn = slave(Stream(Vec(dataType, lanes)))
    
    // Read Interface for Compute Unit
    val readAddr = in UInt(log2Up(depth / lanes) bits)
    val readData = out Vec(dataType, lanes)
    
    // Handshake
    val nextTile = in Bool()   // Pulse from Compute to say "I'm done with this tile"
    val tileReady = out Bool() // High when the current compute bank is full and ready
  }
  
  val memSize = depth / lanes
  
  // Two internal memories (Ping and Pong)
  val memPing = Mem(Vec(dataType, lanes), memSize)
  val memPong = Mem(Vec(dataType, lanes), memSize)
  
  // Pointers/States
  val loadBank = RegInit(False)    // False = Ping, True = Pong
  val computeBank = RegInit(False) // False = Ping, True = Pong
  
  val pingFull = RegInit(False)
  val pongFull = RegInit(False)
  
  // Compute Bank Output Logic
  io.tileReady := (computeBank === False) ? pingFull | pongFull
  val readDataPing = memPing.readAsync(io.readAddr)
  val readDataPong = memPong.readAsync(io.readAddr)
  io.readData := (computeBank === False) ? readDataPing | readDataPong
  
  // Load Process Logic
  val loadCounter = Counter(memSize)
  val currentLoadBankFull = (loadBank === False) ? pingFull | pongFull
  
  io.streamIn.ready := !currentLoadBankFull
  
  val loadDone = io.streamIn.valid && !currentLoadBankFull && loadCounter.willOverflowIfInc
  
  when(io.streamIn.valid && !currentLoadBankFull) {
    // Write data to the correct memory
    when(loadBank === False) {
      memPing.write(loadCounter.value, io.streamIn.payload)
    } otherwise {
      memPong.write(loadCounter.value, io.streamIn.payload)
    }
    loadCounter.increment()
  }
  
  // State management for Ping bank
  when(io.nextTile && computeBank === False) {
    pingFull := False
  } elsewhen(loadDone && loadBank === False) {
    pingFull := True
  }
  
  // State management for Pong bank
  when(io.nextTile && computeBank === True) {
    pongFull := False
  } elsewhen(loadDone && loadBank === True) {
    pongFull := True
  }
  
  // Pointer updates
  when(loadDone) {
    loadBank := !loadBank
  }
  when(io.nextTile) {
    computeBank := !computeBank
  }
}
