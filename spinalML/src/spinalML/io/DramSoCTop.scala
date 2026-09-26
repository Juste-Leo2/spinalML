// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.nn.Accelerator
import spinalML.dtypes.FloatML
import spinalML.Target
import spinalML.memory.DdrAdapter
import spinalML.memory.litedram.LiteDramAxiBridge

/**
 * DRAM-backed UART SoC top: Accelerator + DdrAdapter + LiteDRAM core +
 * AxiReadMem + UartBridge + UartRx/UartTx.
 *
 * Same host protocol as `UartSoC` (docs/uart_bridge.md): the host pre-loads
 * image/weights with 'W', programs CSR, then 'R' drains logits. The only
 * visible differences are the extra `ddram_*` pads and DRAM-sized maps.
 *
 * Reset policy: the SoC (including CSR) is held in reset until the
 * LiteDRAM core reports `init_done` (JEDEC init played in gateware, no
 * BIOS). `init_done` is quasi-static: double-flopped into the board
 * domain before joining the reset tree.
 *
 * Ports: implicit clock (board clk), reset_n (active low), uart_rx,
 * uart_tx, plus the DDR3 pads below (Tang Primer 20K footprint).
 */
class DramSoCTop[T <: Data](
  val acceleratorFactory: () => Accelerator[T],
  val clkFreq: BigInt = 27000000,
  val baudRate: BigInt = 115200,
  val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4),
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000,
  val outCount: Int   = 10,
  val version: Int    = 0x01,
  val target: Target  = Target.FPGA()
) extends Component {
  val io = new Bundle {
    val resetN = in(Bool())
    val uartRx = in(Bool())
    val uartTx = out(Bool())

    // DDR3 pads (see dram/configs/tang-primer-20k.yml + boards pinout).
    val ddram_a       = out Bits(14 bits)
    val ddram_ba      = out Bits(3 bits)
    val ddram_ras_n   = out Bool()
    val ddram_cas_n   = out Bool()
    val ddram_we_n    = out Bool()
    val ddram_cs_n    = out Bool()
    val ddram_dm      = out Bits(2 bits)
    val ddram_dq      = inout(Analog(Bits(16 bits)))
    val ddram_dqs_p   = inout(Analog(Bits(2 bits)))
    val ddram_dqs_n   = inout(Analog(Bits(2 bits)))
    val ddram_clk_p   = out Bool()
    val ddram_clk_n   = out Bool()
    val ddram_cke     = out Bool()
    val ddram_odt     = out Bool()
    val ddram_reset_n = out Bool()
  }

  // FPGA-only top: LiteDRAM has no ASIC mapping, and Gowin BOOT power-on
  // reset has no ASIC equivalent (same rationale as UartSoC).
  require(!target.isAsic, "DramSoCTop is FPGA-only (LiteDRAM core targets Gowin GW2A)")

  // Forward declaration: driven from the SoC area below, sampled here.
  val initDoneRaw = Bool()

  // Power-On Reset + DRAM-init sampling, both in a BOOT domain (no reset
  // input port). The init flops MUST NOT live in `cd`: they have to
  // progress while `cd` is still held in reset, otherwise reset never
  // releases (deadlock). BOOT also keeps the top-level port list minimal
  // (clk / resetN / uart / ddram_* only, no extra reset pin).
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
  // init_done is quasi-static: double-flopped into the board domain before
  // joining the reset tree (reset deassertion stays synchronous).
  val initSync1 = new ClockingArea(bootClockDomain) {
    val s0 = Reg(Bool()) init(False)
    val s1 = Reg(Bool()) init(False)
    s0 := initDoneRaw
    s1 := s0
  }.s1

  val reset = porActive || !io.resetN || !initSync1

  val cd = ClockDomain(
    clock = clockDomain.clock,
    reset = reset,
    config = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = HIGH)
  )

  val soc = new ClockingArea(cd) {
    val acc = acceleratorFactory()

    val outElemBits = acc.io.outStream.stream.payload(0).getBitsWidth
    require(outElemBits == 8,
      s"DramSoCTop only serializes 8-bit output elements over UART (got $outElemBits bits) — see docs/uart_bridge.md")

    val rx = new UartRx(clkFreq, baudRate)
    val tx = new UartTx(clkFreq, baudRate)
    val mem = new DdrAdapter(axiConfig, imgBase, weightBase)
    val dram = new LiteDramAxiBridge(axiConfig)
    val bridge = new UartBridge(
      outCount = outCount,
      wordWidth = axiConfig.dataWidth,
      version = version
    )

    // DRAM core: board reset in (clock mapped to this domain inside the
    // bridge), internal PLL, init status out.
    // (initSync assignment stays in the root domain, outside this area.)
    dram.io.reset_n := io.resetN

    // AXI path: accelerator -> DdrAdapter -> LiteDRAM core.
    mem.io.axi.ar <> acc.io.axiMaster.ar
    mem.io.axi.r  <> acc.io.axiMaster.r
    mem.io.axi.aw <> acc.io.axiMaster.aw
    mem.io.axi.w  <> acc.io.axiMaster.w
    mem.io.axi.b  <> acc.io.axiMaster.b

    dram.io.axi.ar <> mem.extIo.ddrMaster.ar
    dram.io.axi.r  <> mem.extIo.ddrMaster.r
    dram.io.axi.aw <> mem.extIo.ddrMaster.aw
    dram.io.axi.w  <> mem.extIo.ddrMaster.w
    dram.io.axi.b  <> mem.extIo.ddrMaster.b

    // Control bus: bridge (master) <-> accelerator AXI-lite slave
    bridge.io.csr <> acc.io.ctrlBus

    // Host write port (through the DRAM adapter fencing).
    mem.io.wrEnable := bridge.io.wrEnable
    mem.io.wrAddr   := bridge.io.wrAddr
    mem.io.wrData   := bridge.io.wrData
    mem.io.wrStrb   := bridge.io.wrStrb

    // UART wiring (1-cycle pulse handshake, mirror of top.v)
    rx.io.rx := io.uartRx
    bridge.io.rx.valid  := rx.io.valid
    bridge.io.rx.payload := rx.io.data
    tx.io.start := bridge.io.tx.valid
    tx.io.data  := bridge.io.tx.payload
    bridge.io.tx.ready := tx.io.ready
    io.uartTx := tx.io.tx

    // DDR3 pads to the top level.
    io.ddram_a := dram.io.a
    io.ddram_ba := dram.io.ba
    io.ddram_ras_n := dram.io.ras_n
    io.ddram_cas_n := dram.io.cas_n
    io.ddram_we_n := dram.io.we_n
    io.ddram_cs_n := dram.io.cs_n
    io.ddram_dm := dram.io.dm
    io.ddram_dq := dram.io.dq
    io.ddram_dqs_p := dram.io.dqs_p
    io.ddram_dqs_n := dram.io.dqs_n
    io.ddram_clk_p := dram.io.clk_p
    io.ddram_clk_n := dram.io.clk_n
    io.ddram_cke := dram.io.cke
    io.ddram_odt := dram.io.odt
    io.ddram_reset_n := dram.io.ddr_reset_n

    // Output stream: one FP8 byte per logit (mirrors UartSoC).
    val outLanes = acc.io.outStream.lanes
    def toByte(elem: Data): Bits = elem match {
      case f: FloatML => (f.sign ## f.exponent ## f.mantissa).asBits
      case b          => b.asBits.resize(8)
    }

    if (outLanes == 1) {
      val outByte = toByte(acc.io.outStream.stream.payload(0))
      bridge.io.outStream.valid := acc.io.outStream.stream.valid
      bridge.io.outStream.payload := outByte
      acc.io.outStream.stream.ready := bridge.io.outStream.ready
    } else {
      val laneIdx = Reg(UInt(log2Up(outLanes) bits)) init 0
      val outBytes = Vec(Bits(8 bits), outLanes)
      for (i <- 0 until outLanes) {
        outBytes(i) := toByte(acc.io.outStream.stream.payload(i))
      }
      bridge.io.outStream.valid := acc.io.outStream.stream.valid
      bridge.io.outStream.payload := outBytes(laneIdx)

      when(bridge.io.outStream.fire) {
        when(laneIdx === U(outLanes - 1, log2Up(outLanes) bits)) {
          laneIdx := 0
        } otherwise {
          laneIdx := laneIdx + 1
        }
      }
      acc.io.outStream.stream.ready := bridge.io.outStream.ready && (laneIdx === U(outLanes - 1, log2Up(outLanes) bits))
    }

    // Status sources
    bridge.io.statusArValid := acc.io.axiMaster.ar.valid
    bridge.io.statusRValid  := acc.io.axiMaster.r.valid
    bridge.io.accBusy := acc.io.busy
    bridge.io.accDone := acc.io.done
  }

  // Root-domain sampling of the DRAM init flag (see note above).
  initDoneRaw := soc.dram.io.init_done
}
