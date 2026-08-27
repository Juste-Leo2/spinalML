package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite._
import spinalML.tensors.Tensor

/**
 * Top-level wrapper that converts a generic `Sequential` model into a complete
 * System-on-Chip (SoC) ready hardware accelerator.
 * 
 * It automatically exposes two standard memory-mapped buses:
 * - AXI4 Master (High Speed): For fetching features and weights directly from DDR.
 * - AXI4-Lite Slave (Control): For the CPU to configure registers and start the inference.
 */
class Accelerator[T <: Data](
  val dataType: HardType[T],
  val inputShape: Seq[Int],
  val modelSpec: Seq[spinalML.nn.LayerSpec],
  val axiConfig: Axi4Config,
  // Phase-2a weight residency: instantiates the run-mode CSR block (0x10
  // MODE / 0x14 RELOAD) and the underlying control plane in Sequential.
  val weightResidencyCSR: Boolean = true
) extends Component {

  val axiLiteConfig = AxiLite4Config(addressWidth = 8, dataWidth = 32)

  // Cast interne invisible pour l'utilisateur
  val globalDataType = dataType.asInstanceOf[HardType[Data]]

  // 1. Instantiate the neural network datapath first to infer its output shape
  val model = Sequential(globalDataType, inputShape, modelSpec, axiConfig,
    weightResidency = weightResidencyCSR)
  
  val io = new Bundle {
    // High-speed Master for DDR access
    val axiMaster = master(Axi4(axiConfig))
    
    // Low-speed Slave for CPU configuration
    val ctrlBus = slave(AxiLite4(axiLiteConfig))
    
    // Output Stream for the final result
    val outStream = master(cloneOf(model.io.outStream))
  }
  
  // 2. Map the AXI4 Master
  // We connect the Read channels. Write channels are grounded since we only infer (read-only DDR).
  io.axiMaster.ar << model.io.axiMaster.ar
  model.io.axiMaster.r << io.axiMaster.r
  
  io.axiMaster.aw.valid := False
  io.axiMaster.aw.payload.assignDontCare()
  io.axiMaster.w.valid := False
  io.axiMaster.w.payload.assignDontCare()
  io.axiMaster.b.ready := False
  
  // 3. Map the final output stream
  io.outStream <> model.io.outStream
  
  // 4. Create the AXI4-Lite Control Registers
  val ctrlFactory = new AxiLite4SlaveFactory(io.ctrlBus)
  
  // Register 0x00: Control
  // Bit 0: Start inference (trigger)
  val startPending = RegInit(False)
  ctrlFactory.onWrite(0x00) {
    // Trigger inference. We hold the request until the datapath accepts it.
    startPending := True
  }
  
  val startEvent = Event
  startEvent.valid := startPending
  when(startEvent.fire) {
    startPending := False
  }
  
  model.io.start << startEvent
  
  // Register 0x04: Status
  // Bit 0: Done (We can read it if we want, currently tied to outStream valid)
  // Bit 1: Busy — an inference is in flight (START accepted, output frame
  //        not yet complete). Phase-3 S1.
  // Bit 2: RUN state (mirror of 0x1C bit0).
  ctrlFactory.read(io.outStream.stream.valid, 0x04, 0)
  ctrlFactory.read(model.io.busy, 0x04, 1)

  // Register 0x08: Image Base Address. Under RUN auto-advance the MODEL sees
  // the CSR base plus an internal frame cursor (kept in a plain register — the
  // CSR register itself is factory-driven and must not be ticked from
  // arbitrary when-closures): readback 0x08 therefore stays the host-set base,
  // while the effective access point walks base + frameCount × imageBytes.
  val imgAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), 0x08, 0) init(0)

  // Register 0x0C: Weights Base Address
  val weightsAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), 0x0C, 0) init(0)
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
  val runReg = ctrlFactory.createReadAndWrite(UInt(8 bits), 0x1C, 0) init(0)
  val runActive = runReg(0) // sampled at top level (see comment above)
  val imageBytesAcc = (globalDataType().getBitsWidth / 8) * inputShape.product

  val tileCntReg = Reg(UInt(32 bits)) init(0)
  val imgBaseOffset = Reg(UInt(axiConfig.addressWidth bits)) init(0)
  val frameDone = model.io.done
  when(frameDone) {
    tileCntReg := tileCntReg + 1
    when(runActive) {
      // Auto-advance: re-fire START and slide the image cursor forward.
      startPending := True
      imgBaseOffset := imgBaseOffset + imageBytesAcc
    }
  }
  ctrlFactory.read(tileCntReg, 0x18, 0)
  ctrlFactory.read(runActive, 0x04, 2)

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
    val runModeReg = ctrlFactory.createReadAndWrite(UInt(8 bits), 0x10, 0) init(0)
    model.io.weightResident.foreach(_ := runModeReg(0))
    model.io.weightPrefetch.foreach(_ := runModeReg(1))

    val reloadShot = RegInit(False)
    ctrlFactory.onWrite(0x14) {
      reloadShot := True
    }
    when(reloadShot) {
      reloadShot := False
    }
    model.io.weightReload.foreach(_ := reloadShot)
  }
}
