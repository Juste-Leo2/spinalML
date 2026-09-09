// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.nn.Accelerator
import spinalML.dtypes.FloatML

/**
 * UART SoC top: Accelerator + AxiReadMem + UartBridge + UartRx/UartTx,
 * mirroring the reference top.v (reset_n inverted, 27 MHz @ 115200 baud).
 *
 * Ports: clk (global), reset_n (active low), uart_rx, uart_tx.
 * Inference: host pre-loads image/weights with 'W', programs CSR (0x00 start
 * / 0x08 image base / 0x0C weights base), then 'R' drains outCount logits
 * (FP8: {sign, exponent, mantissa}) — see docs/uart_bridge.md.
 */
class UartSoC[T <: Data](
  val acceleratorFactory: () => Accelerator[T],
  val clkFreq: BigInt = 27000000,
  val baudRate: BigInt = 115200,
  val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4),
  val memoryWords: Int = 4096,
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000,
  val outCount: Int   = 10,
  val version: Int    = 0x01
) extends Component {
  val io = new Bundle {
    val resetN = in(Bool())
    val uartRx = in(Bool())
    val uartTx = out(Bool())
  }

  // Power-On Reset: bitstream boot initialization without creating any external reset port
  val bootClockDomain = ClockDomain(
    clock = clockDomain.clock,
    config = ClockDomainConfig(resetKind = BOOT)
  )
  val porActive = new ClockingArea(bootClockDomain) {
    val counter = Reg(UInt(8 bits)) init(0)
    val active = counter =/= 255
    when(active) {
      counter := counter + 1
    }
  }.active

  val reset = porActive || !io.resetN
  val cd = ClockDomain(
    clock = clockDomain.clock,
    reset = reset,
    config = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = HIGH)
  )

  val soc = new ClockingArea(cd) {
    val acc = acceleratorFactory()

    val rx = new UartRx(clkFreq, baudRate)
    val tx = new UartTx(clkFreq, baudRate)
    val mem = new AxiReadMem(axiConfig, memoryWords, imgBase, weightBase)
    val bridge = new UartBridge(
      outCount = outCount,
      wordWidth = axiConfig.dataWidth,
      version = version
    )

    // AXI master (read only): accelerator <-> internal BRAM
    mem.io.axi.ar <> acc.io.axiMaster.ar
    mem.io.axi.r  <> acc.io.axiMaster.r
    acc.io.axiMaster.aw.ready := mem.io.axi.aw.ready
    acc.io.axiMaster.w.ready  := mem.io.axi.w.ready
    acc.io.axiMaster.b.valid  := mem.io.axi.b.valid
    acc.io.axiMaster.b.payload := mem.io.axi.b.payload

    // Control bus: bridge (master) <-> accelerator AXI-lite slave
    bridge.io.csr <> acc.io.ctrlBus

    // BRAM write port
    mem.io.wrEnable := bridge.io.wrEnable
    mem.io.wrAddr   := bridge.io.wrAddr
    mem.io.wrData   := bridge.io.wrData

    // UART wiring (1-cycle pulse handshake, mirror of top.v)
    rx.io.rx := io.uartRx
    bridge.io.rx.valid  := rx.io.valid
    bridge.io.rx.payload := rx.io.data
    tx.io.start := bridge.io.tx.valid
    tx.io.data  := bridge.io.tx.payload
    bridge.io.tx.ready := tx.io.ready
    io.uartTx := tx.io.tx

    // Output stream: one FP8 byte per logit
    val outElem = acc.io.outStream.stream.payload(0)
    val outByte = outElem match {
      case f: FloatML => (f.sign ## f.exponent ## f.mantissa).asBits
      case b          => b.asBits.resize(8)
    }
    bridge.io.outStream.valid := acc.io.outStream.stream.valid
    bridge.io.outStream.payload := outByte
    acc.io.outStream.stream.ready := bridge.io.outStream.ready

    // Status sources
    bridge.io.statusArValid := acc.io.axiMaster.ar.valid
    bridge.io.statusRValid  := acc.io.axiMaster.r.valid
    bridge.io.accBusy := acc.io.busy
    bridge.io.accDone := acc.io.done
  }
}
