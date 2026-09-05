// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.nn

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8
import spinalML.nn.{Accelerator, Conv2D}

class AcceleratorFormal extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 4)
  // Minimal 1x1 1-channel Conv2D: 1 element image, 1 element weight -> instant flow completion
  val spec = Seq(Conv2D(inChannels = 1, outChannels = 1, kernelSize = 1))

  val dut = FormalDut(new Accelerator(
    dataType = I8(),
    inputShape = Seq(1, 1, 1),
    modelSpec = spec,
    axiConfig = axiConfig
  ))

  // Drive CSR control inputs symbolically
  anyseq(dut.io.ctrlBus.aw.valid)
  anyseq(dut.io.ctrlBus.aw.payload.addr)
  dut.io.ctrlBus.aw.payload.prot := 0

  anyseq(dut.io.ctrlBus.w.valid)
  anyseq(dut.io.ctrlBus.w.payload.data)
  dut.io.ctrlBus.w.payload.strb := 0xF

  anyseq(dut.io.ctrlBus.b.ready)

  anyseq(dut.io.ctrlBus.ar.valid)
  anyseq(dut.io.ctrlBus.ar.payload.addr)
  dut.io.ctrlBus.ar.payload.prot := 0

  anyseq(dut.io.ctrlBus.r.ready)

  // Tie AXI Master data to zero to eliminate arithmetic bit-blasting
  anyseq(dut.io.axiMaster.ar.ready)
  anyseq(dut.io.axiMaster.r.valid)
  anyseq(dut.io.axiMaster.r.payload.last)
  dut.io.axiMaster.r.payload.data := 0
  dut.io.axiMaster.r.payload.resp := 0
  dut.io.axiMaster.r.payload.id := 0

  anyseq(dut.io.outStream.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  // Standard AXI-Lite input stability assumptions
  when(pastValid() && past(dut.io.ctrlBus.aw.valid) && !past(dut.io.ctrlBus.aw.ready)) {
    assume(dut.io.ctrlBus.aw.valid)
    assume(dut.io.ctrlBus.aw.payload.addr === past(dut.io.ctrlBus.aw.payload.addr))
  }
  when(pastValid() && past(dut.io.ctrlBus.w.valid) && !past(dut.io.ctrlBus.w.ready)) {
    assume(dut.io.ctrlBus.w.valid)
    assume(dut.io.ctrlBus.w.payload.data === past(dut.io.ctrlBus.w.payload.data))
  }

  // Pull internal state variables from Accelerator
  val startPending   = dut.startPending.pull()
  val runActive      = dut.runActive.pull()
  val tileCntReg     = dut.tileCntReg.pull()
  val imgBaseOffset  = dut.imgBaseOffset.pull()
  val frameDone      = dut.frameDone.pull()
  val startEventFire = dut.startEvent.fire.pull()

  // ==========================================
  // SAFETY PROPERTIES (SoC Control Contracts)
  // ==========================================

  // 1. DDR Read-Only Invariant: AW and W channels must NEVER be active
  assert(dut.io.axiMaster.aw.valid === False, "Accelerator illegally attempted DDR write address")
  assert(dut.io.axiMaster.w.valid === False, "Accelerator illegally attempted DDR write data")
  assert(dut.io.axiMaster.b.ready === False, "Accelerator illegally asserted write response ready")

  // 2. Start Handshake: Once startEvent fires into the datapath, startPending must clear
  when(pastValid() && past(startEventFire)) {
    assert(!startPending, "startPending stayed active after model accepted start")
  }

  // 3. Frame Counter Integrity: TILE_CNT increments if and only if frameDone pulses
  when(pastValid() && past(frameDone)) {
    assert(tileCntReg === past(tileCntReg) + 1, "tileCntReg failed to increment on frameDone")
  }
  when(pastValid() && !past(frameDone)) {
    assert(tileCntReg === past(tileCntReg), "tileCntReg changed spuriously without frameDone")
  }

  // 4. Auto-Advance RUN Mode Contract (CSR 0x1C):
  // When frameDone occurs under RUN=1: startPending re-asserts and image cursor advances.
  when(pastValid() && past(frameDone) && past(runActive)) {
    assert(startPending, "Auto-advance failed to re-arm startPending under RUN=1")
    assert(imgBaseOffset === past(imgBaseOffset) + dut.imageBytesAcc, "Image offset failed to advance by imageBytesAcc")
  }

  // When frameDone occurs under STOP (RUN=0): image cursor stays frozen.
  when(pastValid() && past(frameDone) && !past(runActive)) {
    assert(imgBaseOffset === past(imgBaseOffset), "Image offset mutated after STOP was issued")
  }

  // ==========================================
  // REACHABILITY (Liveness / Cover)
  // ==========================================

  // 1. Start event can fire
  cover(startEventFire)

  // 2. Continuous RUN streaming can be enabled
  cover(runActive)
}

object AcceleratorFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(6)
      .withTimeout(120)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AcceleratorFormal, "accelerator_formal")
  }
}
