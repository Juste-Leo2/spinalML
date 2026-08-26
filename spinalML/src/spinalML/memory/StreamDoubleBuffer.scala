package spinalML.memory

import spinal.core._
import spinal.lib._

/**
 * StreamDoubleBuffer: A generic double-buffering component.
 * It reads from a Stream into Ping/Pong memory banks (BRAM).
 * Exposes a random-access read interface for the computation unit.
 * Handshake interface (nextTile, tileReady) controls the Ping/Pong switching.
 *
 * `io.reArm` is the command-boundary pulse (callers wire it to their
 * producing DMA's cmd.fire): it returns every bank pointer / full flag to
 * its power-on state so a back-to-back command can never observe a full
 * flag left over by the previous command. Without it, ping/pong parity
 * survives across commands and inference N+1 starts by consuming a stale
 * tile (inter-start corruption).
 *
 * `io.freeze` (optional port, weight-residency control plane — Phase 2a):
 * while asserted it NEUTRALISES nextTile — the full flag of the consumed
 * bank is preserved and the compute pointer stops flipping, so the tile
 * just delivered stays visible forever: every later pass re-reads the SAME
 * bank contents through its streamer (weights resident on chip, zero DDR
 * traffic). An actual reArm pulse always wins (last-assignment-wins below),
 * which makes a RELOAD underneath residency behave exactly like today's
 * normal fetch-and-flip pass.
 */
case class StreamDoubleBuffer[T <: Data](dataType: HardType[T], depth: Int, lanes: Int,
                                         enableFreezePort: Boolean = false) extends Component {
  val io = new Bundle {
    // Input Stream
    val streamIn = slave(Stream(Vec(dataType, lanes)))

    // Read Interface for Compute Unit
    val readAddr = in UInt(log2Up(depth / lanes) bits)
    val readData = out Vec(dataType, lanes)

    // Handshake
    val nextTile = in Bool()   // Pulse from Compute to say "I'm done with this tile"
    val tileReady = out Bool() // High when the current compute bank is full and ready

    // Command-boundary re-arm pulse (see class doc)
    val reArm = in Bool()

    // Residency hold (Phase 2a) — neutralises nextTile when asserted.
    // (Named residentHold: `freeze` collides with spinal.core.Data#freeze.)
    val residentHold = if (enableFreezePort) Some(in Bool()) else None
  }

  def freezeNow: Bool = io.residentHold.getOrElse(False)
  
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
  val readDataPing = memPing.readSync(io.readAddr)
  val readDataPong = memPong.readSync(io.readAddr)
  
  // Since readSync adds 1 cycle latency, we should also delay the bank selection signal
  // by 1 cycle to ensure we read from the correct bank.
  val computeBankDelayed = RegNext(computeBank)
  io.readData := (computeBankDelayed === False) ? readDataPing | readDataPong
  
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
  when(io.nextTile && !freezeNow && computeBank === False) {
    pingFull := False
  } elsewhen(loadDone && loadBank === False) {
    pingFull := True
  }

  // State management for Pong bank
  when(io.nextTile && !freezeNow && computeBank === True) {
    pongFull := False
  } elsewhen(loadDone && loadBank === True) {
    pongFull := True
  }

  // Pointer updates
  when(loadDone) {
    loadBank := !loadBank
  }
  when(io.nextTile && !freezeNow) {
    computeBank := !computeBank
  }

  // Command-boundary re-arm: last-assignment-wins, so this overrides any of
  // the updates above on the cycle a new command is accepted.
  when(io.reArm) {
    loadBank := False
    computeBank := False
    pingFull := False
    pongFull := False
    loadCounter.clear()
  }
}
