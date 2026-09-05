// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor
import spinalML.ops.repack
import spinalML.ops.RepackOp

/**
 * FetchRequest specifies where to read and how much.
 * length: Number of AXI beats minus 1 (AXI4-style encoding extended past the
 * 8-bit bus limit). 0 means 1 beat. The DMAReader internally splits requests
 * into chained INCR bursts (at most maxBurstBeats beats each, never crossing
 * a 4 KiB boundary), so tensors of up to 64K beats are supported.
 */
case class FetchRequest(addressWidth: Int) extends Bundle {
  val address = UInt(addressWidth bits)
  val length  = UInt(16 bits)
}

/**
 * DMAReader: An AXI4-Master module that fetches data from DDR4 
 * and automatically adapts the physical bus width to the requested ML Tensor lanes.
 */
case class DMAReader[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  outLanes: Int,
  axiConfig: Axi4Config,
  /** Max AXI beats per INCR burst. Lower it in formal/sim benches so that
    * burst-splitting paths are reachable within short verification windows. */
  val maxBurstBeats: Int = 256,
  /** Emit EXACTLY shape.product elements: see the trim stage below. Weight /
    * bias readers enable this; DMAReader2D keeps its own row trim. */
  val trimToElements: Boolean = false,
  /** Use the flushable structured gearbox instead of the plain width
    * adapter, and gate command acceptance on its drain state. Weight/bias
    * readers enable this: their regions can end mid-group, leaving stale
    * partial groups that would phase-shift every back-to-back command.
    * DMAReader2D keeps the legacy adapter (image rows are exact multiples). */
  val flushableGearbox: Boolean = false
) extends Component {

  val axiLanes = axiConfig.dataWidth / dataType.getBitsWidth
  require(axiConfig.dataWidth % dataType.getBitsWidth == 0, "AXI data width must be a multiple of the data type width")

  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest(axiConfig.addressWidth)))
    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(dataType, shape, outLanes))
  }

  // ==========================================
  // 1. AR (Address Read) Channel Handshake
  // ==========================================
  // Requests longer than the AXI4 arlen limit (256 beats) are split into
  // chained INCR bursts. Bursts are additionally clipped at 4 KiB boundaries,
  // as required by the AXI4 protocol (no burst may cross a 4K edge): the next
  // burst resumes right after the boundary, preserving element order.
  val bytesPerBeat = axiConfig.dataWidth / 8

  // 17 bits so that length = 0xFFFF (+1 beat = 65536) does not overflow to zero
  val remaining   = Reg(UInt(17 bits)).init(0)               // beats not yet requested
  val burstRemain = Reg(UInt(log2Up(maxBurstBeats + 1) bits)).init(0) // beats not yet received
  val addrReg     = Reg(UInt(axiConfig.addressWidth bits)).init(0)

  // With a flushable gearbox, accept a new command ONLY once its tail has
  // drained: flushing earlier would truncate the previous tensor. Once
  // empty, fire still precedes any read data of the new command.
  val baseReady = (remaining === 0) && (burstRemain === 0)
  val gearboxEmpty = Bool() // assigned in section 3, after gearbox creation
  io.cmd.ready := (if (flushableGearbox) baseReady && gearboxEmpty else baseReady)

  when(io.cmd.fire) {
    addrReg := io.cmd.address
    remaining := io.cmd.length +^ 1
  }

  val offsetInPage      = addrReg(0, 12 bits)
  val bytesToBoundary   = (U(4096, 13 bits) - offsetInPage.resize(13 bits))
  val beatsToBoundary   = (bytesToBoundary >> log2Up(bytesPerBeat)).resize(16 bits)
  val burstLen          = remaining.min(U(maxBurstBeats, 17 bits)).min(beatsToBoundary.max(1).resize(17 bits))
  // Strict serialization: issue the next burst only once the previous one has
  // fully drained. Allowing an AR to overlap the tail of the previous burst
  // would make the burstRemain reload race against its last decrements.
  io.axiMaster.ar.valid := (remaining =/= 0) && (burstRemain === 0)
  io.axiMaster.ar.addr  := addrReg
  io.axiMaster.ar.len   := (burstLen - 1).resize(8 bits)
  io.axiMaster.ar.size  := log2Up(bytesPerBeat)
  io.axiMaster.ar.burst := B"01" // INCR burst mode (linear memory fetch)

  when(io.axiMaster.ar.fire) {
    addrReg := addrReg + (burstLen << log2Up(bytesPerBeat)).resize(axiConfig.addressWidth)
    remaining := remaining - burstLen
    burstRemain := burstLen.resized
  }

  // Default tied-off AXI4 signals
  io.axiMaster.ar.id    := 0
  io.axiMaster.ar.prot  := 0
  io.axiMaster.ar.cache := 0
  io.axiMaster.ar.lock  := 0
  io.axiMaster.ar.qos   := 0
  io.axiMaster.ar.region := 0

  // ==========================================
  // 2. R (Read Data) Channel Handshake -> Raw Tensor
  // ==========================================
  val axiRawTensor = Tensor(dataType, shape, axiLanes)

  axiRawTensor.stream.valid := io.axiMaster.r.valid && (burstRemain =/= 0)
  io.axiMaster.r.ready      := axiRawTensor.stream.ready

  when(io.axiMaster.r.valid && io.axiMaster.r.ready) {
    burstRemain := burstRemain - 1
  }
  
  // Convert physical AXI bits into ML DataType array
  for (i <- 0 until axiLanes) {
    val slice = io.axiMaster.r.data(i * dataType.getBitsWidth, dataType.getBitsWidth bits)
    axiRawTensor.stream.payload(i).assignFromBits(slice)
  }

  // ==========================================
  // 3. Internal Gearbox (Repack)
  // ==========================================
  // Automatically adapt the raw physical AXI stream to the desired ML lanes.
  val gearboxOps = scala.collection.mutable.ArrayBuffer[RepackOp[_]]()
  val repackedTensor = repack(axiRawTensor, outLanes,
    reArm = if (flushableGearbox) Some(io.cmd.fire) else None,
    created = gearboxOps, withFlush = flushableGearbox)
  gearboxEmpty := (if (gearboxOps.isEmpty) True
                   else gearboxOps.map(_.io.isEmpty).reduce(_ && _))

  // Exact-element trim (see trimToElements): hide everything past the last
  // logical element of this command so the stream ends GROUP-ALIGNED — whole-
  // beat fetches otherwise drag region-padding elements past the tensor end,
  // parking partial groups in the lane gearbox and junk words in downstream
  // exact-size double buffers, which poisons back-to-back commands. The
  // counter restarts at each cmd.fire, which precedes any read data of the
  // new command.
  val trimmedStream = if (!trimToElements) repackedTensor else {
    val total = shape.product
    val sent = Reg(UInt(log2Up(total + outLanes + 1) bits)) init (0)
    when(io.cmd.fire) { sent := 0 }
    val suppressed = sent >= U(total)
    val trimmed = Tensor(dataType, shape, outLanes)
    trimmed.stream.valid := repackedTensor.stream.valid && !suppressed
    repackedTensor.stream.ready := io.outStream.stream.ready || suppressed
    when(trimmed.stream.fire) { sent := sent + U(outLanes) }
    trimmed.stream.payload := repackedTensor.stream.payload
    trimmed
  }

  io.outStream <> trimmedStream
}
