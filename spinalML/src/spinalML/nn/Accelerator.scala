// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite._
import spinalML.tensors.Tensor
import spinalML.memory.{DMAWriter, WriteRequest}
import spinalML.Target

/**
 * Top-level wrapper that converts a generic `Sequential` model into a complete
 * System-on-Chip (SoC) ready hardware accelerator.
 *
 * It automatically exposes two standard memory-mapped buses:
 * - AXI4 Master (High Speed): For fetching features and weights directly from DDR,
 *   and writing back output tensors via DMAWriter.
 * - AXI4-Lite Slave (Control): For the CPU to configure registers and start the inference.
 *
 * @param target Hardware target carried through to `Sequential`
 *               (Phase-1 DDR plumbing, docs/ddr_impl.md). Default Simulation
 *               preserves current CI behavior; pass `Target.FPGA(...)` or
 *               `Target.ASIC(...)` to elaborate for silicon.
 */
class Accelerator[T <: Data](
  val dataType: HardType[T],
  val inputShape: Seq[Int],
  val modelSpec: Seq[spinalML.nn.LayerSpec],
  val axiConfig: Axi4Config,
  // Phase-2a weight residency: instantiates the run-mode CSR block (0x10
  // MODE / 0x14 RELOAD) and the underlying control plane in Sequential.
  val weightResidencyCSR: Boolean = true,
  // Phase-3 activation tiling: image rows per vertical band (see Sequential).
  // <= 0 keeps the legacy full-image behaviour.
  val tileHeight: Int = -1,
  // M3: rows-in-flight bound for the reduction ops (see Sequential.temporal).
  // 0 = legacy full MxN accumulator table; > 0 = windowed row drain.
  val temporal: Int = 0,
  val inLanes: Int = 1,
  val target: Target = Target.Simulation,
  // Phase-2 DDR plumbing (docs/ddr_impl.md): logical memory descriptor.
  // Default = legacy map (on-chip, historical bases, no fit check), so every
  // existing model elaborates exactly as before.
  val memory: MemorySpec = MemorySpec.default
) extends Component {

  val axiLiteConfig = AxiLite4Config(addressWidth = 8, dataWidth = 32)

  // Cast interne invisible pour l'utilisateur
  val globalDataType = dataType.asInstanceOf[HardType[Data]]

  // 1. Instantiate the neural network datapath first to infer its output shape
  val model = Sequential(globalDataType, inputShape, modelSpec, axiConfig,
    weightResidency = weightResidencyCSR, tileHeight = tileHeight, temporal = temporal, inLanes = inLanes,
    target = target)

  // Instantiate DMAWriter for optional DDR write-back of final output tensor
  val dmaWriter = DMAWriter(
    dataType = model.finalType,
    shape = model.finalShape,
    inLanes = model.finalLanes,
    axiConfig = axiConfig
  )

  val outLanesAxi = axiConfig.dataWidth / model.finalType.getBitsWidth
  val totalOutBeats = (model.finalShape.product + outLanesAxi - 1) / outLanesAxi
  
  val io = new Bundle {
    // High-speed Master for DDR access (Read & Write)
    val axiMaster = master(Axi4(axiConfig))
    
    // Low-speed Slave for CPU configuration
    val ctrlBus = slave(AxiLite4(axiLiteConfig))
    
    // Output Stream for the final result
    val outStream = master(cloneOf(model.io.outStream))

    // Accelerator status (UART bridge status byte): busy = inference in
    // flight, done = frame-complete pulse.
    val busy = out(Bool())
    val done = out(Bool())
  }

  // 4. Create the AXI4-Lite Control Registers
  val ctrlFactory = new AxiLite4SlaveFactory(io.ctrlBus)

  // Register 0x20: Output Base Address (for DMAWriter write-back)
  val outAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), CsrMap.OutAddr, 0) init(0)

  // Register 0x24: Output Control (bit 0: writeToDdr enable)
  val outCtrlReg = ctrlFactory.createReadAndWrite(UInt(8 bits), CsrMap.OutCtrl, 0) init(0)
  val writeToDdr = outCtrlReg(0)

  // Register 0x28: DMA Write Status (bit 0: busy, bit 1: done)
  ctrlFactory.read(dmaWriter.io.busy, CsrMap.DmaStatus, 0)
  ctrlFactory.read(dmaWriter.io.done, CsrMap.DmaStatus, 1)

  // Register 0x30: Runtime Dequantization Scale (for Cast layers with runtimeScale = true)
  val initialScaleBits: BigInt = {
    val castLayerOpt = modelSpec.collectFirst { case c: Cast if c.runtimeScale => c }
    castLayerOpt match {
      case Some(c) =>
        c.targetType() match {
          case f: spinalML.dtypes.FloatML =>
            spinalML.utils.MathLUTs.floatEncodeFn(f.expBits, f.mantBits)(c.scales.headOption.getOrElse(1.0))
          case _ => BigInt(1)
        }
      case None => BigInt(0)
    }
  }
  val dequantScaleReg = ctrlFactory.createReadAndWrite(Bits(32 bits), CsrMap.DequantScale, 0) init(B(initialScaleBits, 32 bits))
  model.io.dequantScale.foreach(_ := dequantScaleReg)

  // 2. Map the AXI4 Master
  // Read channels: connected to Sequential model
  io.axiMaster.ar << model.io.axiMaster.ar
  model.io.axiMaster.r << io.axiMaster.r

  // Write channels: routed to DMAWriter when writeToDdr is active, grounded otherwise
  when(writeToDdr) {
    io.axiMaster.aw.valid := dmaWriter.io.axiMaster.aw.valid
    io.axiMaster.aw.payload := dmaWriter.io.axiMaster.aw.payload
    dmaWriter.io.axiMaster.aw.ready := io.axiMaster.aw.ready

    io.axiMaster.w.valid := dmaWriter.io.axiMaster.w.valid
    io.axiMaster.w.payload := dmaWriter.io.axiMaster.w.payload
    dmaWriter.io.axiMaster.w.ready := io.axiMaster.w.ready

    dmaWriter.io.axiMaster.b.valid := io.axiMaster.b.valid
    dmaWriter.io.axiMaster.b.payload := io.axiMaster.b.payload
    io.axiMaster.b.ready := dmaWriter.io.axiMaster.b.ready
  } otherwise {
    io.axiMaster.aw.valid := False
    io.axiMaster.aw.payload.assignDontCare()
    dmaWriter.io.axiMaster.aw.ready := False

    io.axiMaster.w.valid := False
    io.axiMaster.w.payload.assignDontCare()
    dmaWriter.io.axiMaster.w.ready := False

    io.axiMaster.b.ready := False
    dmaWriter.io.axiMaster.b.valid := False
    dmaWriter.io.axiMaster.b.payload.assignDontCare()
  }

  // 3. Map the final output stream
  when(writeToDdr) {
    dmaWriter.io.inStream.stream.valid := model.io.outStream.stream.valid
    dmaWriter.io.inStream.stream.payload := model.io.outStream.stream.payload
    model.io.outStream.stream.ready := dmaWriter.io.inStream.stream.ready

    io.outStream.stream.valid := False
    io.outStream.stream.payload.assignDontCare()
  } otherwise {
    io.outStream.stream.valid := model.io.outStream.stream.valid
    io.outStream.stream.payload := model.io.outStream.stream.payload
    model.io.outStream.stream.ready := io.outStream.stream.ready

    dmaWriter.io.inStream.stream.valid := False
    dmaWriter.io.inStream.stream.payload.assignDontCare()
  }
  
  // Register 0x00: Control
  // Bit 0: Start inference (trigger)
  val startPending = RegInit(False)
  // Sticky completion flag for the DDR path: frameDone is a single-cycle pulse
  // but a host AXI-Lite status read spans many cycles. Cleared by a host START
  // write so a polling driver cannot miss completion.
  val doneSticky = RegInit(False)
  ctrlFactory.onWrite(CsrMap.Start) {
    // Trigger inference. We hold the request until the datapath accepts it.
    startPending := True
    doneSticky := False
  }
  
  val outBytesAcc = totalOutBeats * (axiConfig.dataWidth / 8)
  val outBaseOffset = Reg(UInt(axiConfig.addressWidth bits)) init(0)

  val startEvent = Event
  val dmaCmd = Stream(WriteRequest(axiConfig.addressWidth))
  dmaCmd.address := outAddrReg + outBaseOffset
  dmaCmd.length := U(totalOutBeats - 1, 16 bits)

  when(writeToDdr) {
    startEvent.valid := startPending && dmaCmd.ready
    dmaCmd.valid := startPending && model.io.start.ready
    model.io.start.valid := startPending && dmaCmd.ready
    startEvent.ready := model.io.start.ready && dmaCmd.ready
  } otherwise {
    startEvent.valid := startPending
    model.io.start.valid := startEvent.valid
    startEvent.ready := model.io.start.ready
    dmaCmd.valid := False
  }

  when(startEvent.fire) {
    startPending := False
  }

  dmaWriter.io.cmd << dmaCmd
  
  // Register 0x08: Image Base Address. Under RUN auto-advance the MODEL sees
  // the CSR base plus an internal frame cursor (kept in a plain register — the
  // CSR register itself is factory-driven and must not be ticked from
  // arbitrary when-closures): readback 0x08 therefore stays the host-set base,
  // while the effective access point walks base + frameCount × imageBytes.
  val imgAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), CsrMap.ImgBase, 0) init(0)

  // Register 0x0C: Weights Base Address
  val weightsAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), CsrMap.WeightBase, 0) init(0)
  model.io.weightsBaseAddress := weightsAddrReg

  // ------------------------------------------------------------------
  // Continuous run control (Phase 3, S1)
  //
  // A one-shot START (write 0x00) still runs exactly one inference: this
  // register is purely opt-in for the streaming contract.
  // Register 0x1C RUN: bit0 = 1 enables AUTO-ADVANCE — when the datapath
  //   fires its frame-complete pulse, the accelerator immediately issues
  //   the next START. Together with the image-base cursor below this turns
  //   the accelerator into a frame-streaming engine (video-style): the host
  //   pre-writes N images at consecutive DDR addresses, programs 0x08 ONCE,
  //   sets RUN, writes a single START — inference k reads image k.
  //   SETTING RUN BACK TO 0 stops the engine after the in-flight frame
  //   (no new START is issued; the current inference completes cleanly).
  //   Contract: the host MUST have written image k to DDR before frame k−1
  //   completes (the hardware raises busy on the first START and the
  //   pipeline keeps whatever 0x08 points at).
  // Register 0x18 TILE_CNT: read-only count of completed output frames since
  //   reset (matches each 0x04-bit1 busy falling edge).
  // Race note: the cursor is advanced on the SAME edge as the auto-START,
  //   strictly before any DMA reads the address — the host never observes a
  //   half-advanced frame access.
  val runReg = ctrlFactory.createReadAndWrite(UInt(8 bits), CsrMap.Run, 0) init(0)
  val runActive = runReg(0) // sampled at top level (see comment above)
  val imageBytesAcc = (globalDataType().getBitsWidth / 8) * inputShape.product

  // Phase-2 DDR plumbing: elaboration-time footprint check against the
  // declared capacity (no-op for the legacy default with capacityBytes=None).
  // Frame cursors (imgBaseOffset/outBaseOffset) stay runtime registers —
  // Phase 4 generalizes them with the spill cursor (CSR 0x34 reserved).
  memory.reportFit(imageBytesAcc.toLong, model.totalWeightBytes.toLong, outBytesAcc.toLong)

  val tileCntReg = Reg(UInt(32 bits)) init(0)
  val imgBaseOffset = Reg(UInt(axiConfig.addressWidth bits)) init(0)
  val frameDone = Mux(writeToDdr, dmaWriter.io.done, model.io.done)
  io.busy := Mux(writeToDdr, model.io.busy || dmaWriter.io.busy, model.io.busy)
  io.done := frameDone

  when(frameDone) {
    tileCntReg := tileCntReg + 1
    doneSticky := True
    when(runActive) {
      // Auto-advance: re-fire START and slide the image cursor forward.
      startPending := True
      imgBaseOffset := imgBaseOffset + imageBytesAcc
      when(writeToDdr) {
        outBaseOffset := outBaseOffset + outBytesAcc
      }
    }
  }

  // A host write to 0x08 starts a new image stream: reset the RUN cursor so the
  // next inference reads from the newly programmed base without a hard reset.
  // Placed after the frameDone increment so a same-cycle host write wins.
  ctrlFactory.onWrite(CsrMap.ImgBase) {
    imgBaseOffset := 0
  }

  // A host write to 0x20 starts a new output stream: reset the write-back cursor.
  ctrlFactory.onWrite(CsrMap.OutAddr) {
    outBaseOffset := 0
  }

  // Register 0x04: Status
  // Bit 0: Done (latched frameDone in DDR mode, outStream.valid in stream mode)
  // Bit 1: Busy (model busy || dmaWriter busy)
  // Bit 2: RUN state (mirror of 0x1C bit0)
  ctrlFactory.read(Mux(writeToDdr, doneSticky, io.outStream.stream.valid), CsrMap.Status, 0)
  ctrlFactory.read(io.busy, CsrMap.Status, 1)
  ctrlFactory.read(runActive, CsrMap.Status, 2)
  ctrlFactory.read(tileCntReg, CsrMap.TileCnt, 0)

  model.io.imgBaseAddress := imgAddrReg + imgBaseOffset

  // ------------------------------------------------------------------
  // Weight-residency run-mode control plane (Phase 2a + 2b prefetch)
  //
  // Register 0x10 MODE:
  //   bit0 = WEIGHT_RESIDENT — weight/bias regions are fetched from DDR on
  //          their first use (or one pass after RELOAD) and kept resident;
  //          every later START only repeats the activation/image fetch.
  //   bit1 = PREFETCH_EN (requires bit0) — refresh fetches leave the START
  //          sweep: a RELOAD fires eagerly against reader-ready × loader-
  //          empty, filling the IDLE bank while the held tile is still being
  //          consumed; the consumer switches onto the fresh weights at the
  //          NEXT end-of-pass edge (never mid-stream).
  //   0x00 = STREAM_PER_PASS — today's behaviour: every START re-fetches
  //          everything.
  // Register 0x14 RELOAD: any write pulses a one-shot request so the next
  //   boundary (START, or eager fire when prefetching) re-fetches all
  //   weight/bias regions from the CURRENT 0x0C base.
  // Assumption (documented in docs): the host paces RELOAD requests at most
  // one outstanding per region — BUSY/export may be added later if needed.
  if (weightResidencyCSR) {
    val runModeReg = ctrlFactory.createReadAndWrite(UInt(8 bits), CsrMap.Mode, 0) init(0)
    model.io.weightResident.foreach(_ := runModeReg(0))
    model.io.weightPrefetch.foreach(_ := runModeReg(1))

    val reloadShot = RegInit(False)
    ctrlFactory.onWrite(CsrMap.Reload) {
      reloadShot := True
    }
    when(reloadShot) {
      reloadShot := False
    }
    model.io.weightReload.foreach(_ := reloadShot)
  }
}
