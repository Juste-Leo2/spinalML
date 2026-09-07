// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._

/**
 * AXI-lite CSR stub: same register map as the Accelerator (0x08 imgBase,
 * 0x0C weights base, 0x1C run reg, 0x00 start pulse). Explicit handshake
 * (aw+wr accept, single b response) for test determinism.
 */
case class CsrStub() extends Component {
  import spinal.lib.bus.amba4.axilite._
  val io = new Bundle {
    val ctrl       = slave(AxiLite4(AxiLite4Config(addressWidth = 8, dataWidth = 32)))
    val imgBaseR   = out(UInt(32 bits))
    val weightBaseR = out(UInt(32 bits))
    val runRegR    = out(UInt(8 bits))
    val startPulse = out(Bool())
  }

  val awAddrR = Reg(UInt(8 bits)) init 0
  val wDataR  = Reg(UInt(32 bits)) init 0
  val bValidR = RegInit(False)

  val accept = io.ctrl.aw.valid && io.ctrl.w.valid && !bValidR
  io.ctrl.aw.ready := !bValidR
  io.ctrl.w.ready := !bValidR
  when(accept) {
    awAddrR := io.ctrl.aw.payload.addr
    wDataR := io.ctrl.w.payload.data.asUInt
    bValidR := True
  }
  when(bValidR && io.ctrl.b.ready) {
    bValidR := False
  }
  io.ctrl.b.valid := bValidR
  io.ctrl.b.payload.resp := B"00"

  val imgR = Reg(UInt(32 bits)) init 0
  val wgtR = Reg(UInt(32 bits)) init 0
  val runR = Reg(UInt(8 bits)) init 0
  when(bValidR && io.ctrl.b.ready) {
    switch(awAddrR) {
      is(0x08) { imgR := wDataR }
      is(0x0C) { wgtR := wDataR }
      is(0x1C) { runR := wDataR(7 downto 0) }
    }
  }

  val startPending = RegInit(False)
  when(io.ctrl.aw.valid && io.ctrl.w.valid && !bValidR && (io.ctrl.aw.payload.addr === 0x00)) {
    startPending := True
  }
  when(startPending) { startPending := False }

  io.imgBaseR := imgR
  io.weightBaseR := wgtR
  io.runRegR := runR
  io.startPulse := startPending
}

/**
 * Bridge test component used by the python scenario test (test_uart_bridge.py).
 *
 * In addition to the UART pins it exposes:
 *  - `bram` AXI4 slave: python issues read transactions to verify the 'W'
 *    memory writes (the same BRAM model as the SoC);
 *  - `imgBaseR`/`weightBaseR`/`runRegR`: the CSR stub registers, to verify
 *    the 'C' command without needing the full accelerator;
 *  - `outValidR`: the stub output stream valid, driven by a 0..outCount-1
 *    byte counter enabled by a CSR start write.
 */
case class UartBridgeTestComp(
  clkFreq: BigInt = 27000000,
  baudRate: BigInt = 115200,
  outCount: Int = 10,
  version: Int = 0x01
) extends Component {
  val axiConfig = spinal.lib.bus.amba4.axi.Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val io = new Bundle {
    val rxIn = in(Bool())
    val txOut = out(Bool())

    // Observables for the scenario test
    val rxPulseO  = out(Bool())  // UartRx commit pulse (bridge input)
    val rxDataO   = out(Bits(8 bits))  // UartRx committed byte
    val txFiringO = out(Bool())  // bridge TX request pulse (txStartR)
    // CSR handshake observables (diagnostics)
    val csrAwValidO  = out(Bool())
    val csrAwReadyO  = out(Bool())
    val csrWValidO   = out(Bool())
    val csrWReadyO   = out(Bool())
    val csrBValidO   = out(Bool())
    val csrAwAddrO   = out(UInt(8 bits))
    val csrWDataO    = out(UInt(32 bits))

    val bram = slave(spinal.lib.bus.amba4.axi.Axi4(axiConfig))

    val imgBaseR   = out(UInt(32 bits))
    val weightBaseR = out(UInt(32 bits))
    val runRegR    = out(UInt(8 bits))
    val outValidR  = out(Bool())
  }

  val rx = new UartRx(clkFreq, baudRate)
  val tx = new UartTx(clkFreq, baudRate)
  val bridge = new UartBridge(outCount = outCount, wordWidth = axiConfig.dataWidth, version = version)
  val mem = new AxiReadMem(axiConfig, memoryWords = 4096, imgBase = 0x10000, weightBase = 0x20000)

  io.rxPulseO := rx.io.valid
  io.rxDataO := rx.io.data
  io.txFiringO := bridge.io.tx.valid

  rx.io.rx := io.rxIn
  bridge.io.rx.valid := rx.io.valid
  bridge.io.rx.payload := rx.io.data
  tx.io.start := bridge.io.tx.valid
  tx.io.data := bridge.io.tx.payload
  bridge.io.tx.ready := tx.io.ready
  io.txOut := tx.io.tx

  // BRAM write port from the bridge; independent AXI slave for readback
  mem.io.wrEnable := bridge.io.wrEnable
  mem.io.wrAddr := bridge.io.wrAddr
  mem.io.wrData := bridge.io.wrData
  mem.io.axi.ar <> io.bram.ar
  mem.io.axi.r <> io.bram.r
  mem.io.axi.aw.valid := io.bram.aw.valid
  mem.io.axi.aw.payload := io.bram.aw.payload
  io.bram.aw.ready := mem.io.axi.aw.ready
  mem.io.axi.w.valid := io.bram.w.valid
  mem.io.axi.w.payload := io.bram.w.payload
  io.bram.w.ready := mem.io.axi.w.ready
  io.bram.b.valid := mem.io.axi.b.valid
  io.bram.b.payload := mem.io.axi.b.payload
  mem.io.axi.b.ready := io.bram.b.ready

  // CSR stub: identical register map as the Accelerator (component form,
  // so the master/slave connect follows the UartSoC pattern).
  val csrStub = new CsrStub()
  bridge.io.csr <> csrStub.io.ctrl
  io.csrAwValidO := csrStub.io.ctrl.aw.valid
  io.csrAwReadyO := csrStub.io.ctrl.aw.ready
  io.csrWValidO := csrStub.io.ctrl.w.valid
  io.csrWReadyO := csrStub.io.ctrl.w.ready
  io.csrBValidO := csrStub.io.ctrl.b.valid
  io.csrAwAddrO := csrStub.io.ctrl.aw.payload.addr
  io.csrWDataO := csrStub.io.ctrl.w.payload.data.asUInt
  io.imgBaseR := csrStub.io.imgBaseR
  io.weightBaseR := csrStub.io.weightBaseR
  io.runRegR := csrStub.io.runRegR

  // Stub output stream: 0, 1, ... outCount-1, one byte per fire
  val cnt = Reg(UInt(8 bits)) init 0
  val active = RegInit(False)
  when(csrStub.io.startPulse) { active := True; cnt := 0 }
  bridge.io.outStream.payload := cnt(7 downto 0).asBits
  bridge.io.outStream.valid := active
  io.outValidR := active
  when(active && bridge.io.outStream.ready) {
    cnt := cnt + 1
    when(cnt === (outCount - 1)) { active := False }
  }

  // Status sources for the bridge status byte
  bridge.io.statusArValid := False
  bridge.io.statusRValid := False
  bridge.io.accBusy := active
  bridge.io.accDone := active
}

class UartBridgeTest extends AnyFunSuite {
  test("uart_bridge_toplevel") {
    SpinalConfig(
      headerWithDate = true,
      rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
    ).generateVerilog(UartBridgeTestComp(27000000, 115200))
  }
}
