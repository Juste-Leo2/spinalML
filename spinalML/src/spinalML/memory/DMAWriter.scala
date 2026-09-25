// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor
import spinalML.ops.repack

/**
 * WriteRequest specifies where to write and how much.
 * address: Byte address where the write starts.
 * length: Number of AXI beats minus 1 (0 means 1 beat).
 *         Supports up to 64K beats (16 bits), internally split into INCR bursts
 *         of at most maxBurstBeats beats, respecting 4 KiB page boundaries.
 */
case class WriteRequest(addressWidth: Int) extends Bundle {
  val address = UInt(addressWidth bits)
  val length  = UInt(16 bits)
}

/**
 * DMAWriter: An AXI4-Master write module that drains a streaming Tensor
 * and writes it to DDR/AXI4 memory. Automatically adapts the input Tensor lanes
 * to the physical AXI bus width using the proven repack operator.
 */
case class DMAWriter[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  inLanes: Int,
  axiConfig: Axi4Config,
  maxBurstBeats: Int = 256
) extends Component {

  val bytesPerBeat = axiConfig.dataWidth / 8
  val axiLanes = axiConfig.dataWidth / dataType.getBitsWidth
  require(axiConfig.dataWidth % dataType.getBitsWidth == 0,
    "AXI data width must be a multiple of the data type width")

  val io = new Bundle {
    val cmd       = slave(Stream(WriteRequest(axiConfig.addressWidth)))
    val inStream  = slave(Tensor(dataType, shape, inLanes))
    val axiMaster = master(Axi4WriteOnly(axiConfig))
    val busy      = out(Bool())
    val done      = out(Bool())
  }

  // 1. Adapter: Pack input Tensor lanes to physical AXI bus width
  val totalElements = shape.product
  val elemWidth = dataType.getBitsWidth
  val totalInBeats = (totalElements + inLanes - 1) / inLanes
  val totalAxiBeats = (totalElements + axiLanes - 1) / axiLanes

  val wDataValid = Bool()
  val wDataBits  = Bits(axiConfig.dataWidth bits)
  val wDataReady = Bool()

  if (inLanes == axiLanes) {
    val inBits = Bits(axiConfig.dataWidth bits)
    for (i <- 0 until axiLanes) {
      inBits(i * elemWidth, elemWidth bits) := io.inStream.stream.payload(i).asBits
    }
    wDataBits := inBits
    wDataValid := io.inStream.stream.valid
    io.inStream.stream.ready := wDataReady

  } else if (inLanes < axiLanes) {
    val m = axiLanes / inLanes
    val inW = inLanes * elemWidth
    val buffer = Reg(Bits(axiConfig.dataWidth bits))
    val subIdx = Reg(UInt(log2Up(m) bits)).init(0)
    val inBeatCount = Reg(UInt(log2Up(totalInBeats + 1) bits)).init(0)
    val bufferFull = RegInit(False)

    val inBits = Bits(inW bits)
    for (i <- 0 until inLanes) {
      inBits(i * elemWidth, elemWidth bits) := io.inStream.stream.payload(i).asBits
    }

    io.inStream.stream.ready := !bufferFull

    when(io.inStream.stream.fire) {
      buffer(subIdx * inW, inW bits) := inBits
      val isLastIn = (inBeatCount === totalInBeats - 1)
      when(subIdx === m - 1 || isLastIn) {
        bufferFull := True
        subIdx := 0
      } otherwise {
        subIdx := subIdx + 1
      }

      when(isLastIn) {
        inBeatCount := 0
      } otherwise {
        inBeatCount := inBeatCount + 1
      }
    }

    when(wDataReady && bufferFull) {
      bufferFull := False
    }

    when(io.cmd.fire) {
      bufferFull := False
      subIdx := 0
      inBeatCount := 0
    }

    wDataBits := buffer
    wDataValid := bufferFull

  } else {
    // inLanes > axiLanes (Splitting)
    val k = inLanes / axiLanes
    val outW = axiConfig.dataWidth
    val hold = Reg(Bits(inLanes * elemWidth bits))
    val splitIdx = Reg(UInt(log2Up(k) bits)).init(0)
    val hasData = RegInit(False)

    val inBits = Bits(inLanes * elemWidth bits)
    for (i <- 0 until inLanes) {
      inBits(i * elemWidth, elemWidth bits) := io.inStream.stream.payload(i).asBits
    }

    io.inStream.stream.ready := !hasData || (wDataReady && splitIdx === k - 1)

    when(io.inStream.stream.fire) {
      hold := inBits
      hasData := True
      splitIdx := 0
    } otherwise {
      when(wDataReady && hasData) {
        when(splitIdx === k - 1) {
          hasData := False
        }
        splitIdx := splitIdx + 1
      }
    }

    when(io.cmd.fire) {
      hasData := False
      splitIdx := 0
    }

    wDataBits := hold(splitIdx * outW, outW bits)
    wDataValid := hasData
  }

  // 2. Control Registers & Counters
  // 17 bits so that length = 0xFFFF (+1 beat = 65536) does not overflow to zero
  val remaining   = Reg(UInt(17 bits)).init(0)
  val burstRemain = Reg(UInt(log2Up(maxBurstBeats + 1) bits)).init(0)
  val addrReg     = Reg(UInt(axiConfig.addressWidth bits)).init(0)
  val pendingB    = Reg(UInt(8 bits)).init(0)

  val idle = (remaining === 0) && (burstRemain === 0) && (pendingB === 0)
  io.cmd.ready := idle

  when(io.cmd.fire) {
    addrReg   := io.cmd.address
    remaining := io.cmd.length +^ 1
  }

  // 4 KiB boundary computation
  val offsetInPage    = addrReg(0, 12 bits)
  val bytesToBoundary = (U(4096, 13 bits) - offsetInPage.resize(13 bits))
  val beatsToBoundary = (bytesToBoundary >> log2Up(bytesPerBeat)).resize(16 bits)
  val burstLen        = remaining.min(U(maxBurstBeats, 17 bits)).min(beatsToBoundary.max(1).resize(17 bits))

  // 3. AW (Address Write) Channel
  // Strictly serialized: issue the next burst only once the previous one's W data has fully drained
  io.axiMaster.aw.valid  := (remaining =/= 0) && (burstRemain === 0)
  io.axiMaster.aw.addr   := addrReg
  io.axiMaster.aw.len    := (burstLen - 1).resize(8 bits)
  io.axiMaster.aw.size   := log2Up(bytesPerBeat)
  io.axiMaster.aw.burst  := B"01" // INCR burst
  io.axiMaster.aw.id     := 0
  io.axiMaster.aw.prot   := 0
  io.axiMaster.aw.cache  := 0
  io.axiMaster.aw.lock   := 0
  io.axiMaster.aw.qos    := 0
  io.axiMaster.aw.region := 0

  val awFire = io.axiMaster.aw.fire
  when(awFire) {
    addrReg     := addrReg + (burstLen << log2Up(bytesPerBeat)).resize(axiConfig.addressWidth)
    remaining   := remaining - burstLen
    burstRemain := burstLen.resized
  }

  // 4. W (Write Data) Channel
  val wFire = io.axiMaster.w.fire
  io.axiMaster.w.valid := wDataValid && (burstRemain =/= 0)
  io.axiMaster.w.data  := wDataBits

  // Byte strobes: the transfer is contiguous, so a partial final beat can only
  // be THE last beat of the whole transfer. After the last AW fires `remaining`
  // drops to 0 while `burstRemain` counts the final burst down, so mask the
  // padding bytes of the final beat instead of strobing them all (which would
  // corrupt the bytes adjacent to the tensor).
  val finalBeatElems = totalElements - (totalAxiBeats - 1) * axiLanes
  val finalBeatBytes = (finalBeatElems * elemWidth + 7) / 8
  val fullByteMask = (BigInt(1) << bytesPerBeat) - 1
  val lastByteMask = (BigInt(1) << finalBeatBytes) - 1
  val finalBeat = (remaining === 0) && (burstRemain === 1)
  io.axiMaster.w.strb := Mux(finalBeat,
    B(lastByteMask, bytesPerBeat bits),
    B(fullByteMask, bytesPerBeat bits))
  io.axiMaster.w.last  := (burstRemain === 1)

  wDataReady := io.axiMaster.w.ready && (burstRemain =/= 0)

  when(wFire) {
    burstRemain := burstRemain - 1
  }

  // 5. B (Write Response) Channel
  io.axiMaster.b.ready := True

  val bFire = io.axiMaster.b.fire
  // AW and B may fire in the same cycle (a new burst issued while the previous
  // response arrives); the two effects must cancel instead of clobbering each
  // other, otherwise pendingB underflows on the following response.
  // Saturating decrement: an unsolicited B (sim bring-up transients — never a
  // protocol-correct slave) must not underflow the counter into a permanent
  // cmd.ready wedge. Proof-neutral: DMAWriterFormal assumes B only with
  // pending bursts in flight.
  when(awFire && !bFire) {
    pendingB := pendingB + 1
  } elsewhen (bFire && !awFire && pendingB =/= 0) {
    pendingB := pendingB - 1
  }

  // 6. Status & Done Signals
  io.busy := !idle

  val donePulse = RegInit(False)
  donePulse := False
  when(bFire && pendingB === 1 && remaining === 0 && burstRemain === 0) {
    donePulse := True
  }
  io.done := donePulse
}
