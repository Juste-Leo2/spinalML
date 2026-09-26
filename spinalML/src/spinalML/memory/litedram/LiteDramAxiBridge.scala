// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory.litedram

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

/**
 * Standard-AXI4 to LiteDRAM-streaming-AXI bridge (owns the `LiteDramCore`).
 *
 * The SpinalHDL side (`DdrAdapter.extIo.ddrMaster`) speaks vanilla AXI4;
 * the LiteDRAM frontend expects LiteX streaming sidebands:
 * - Address channels are single-beat: `aw_first`/`aw_last` (and AR) tied high.
 * - Write bursts are framed by `w_first`/`w_last`, tracked here with a beat
 *   counter latched from AW (one outstanding write transaction at a time,
 *   guaranteed by `DdrAdapter` bus ownership).
 * - `dest`/`user` params are unused: tied low (in) / left open (out).
 * - Byte address `addr[31:0]` maps to the core word address `addr[29:3]`
 *   (64-bit words); bits [31:30] must be zero, i.e. the 1GiB window covers
 *   the 128MiB DRAM with aliasing above it (all SpinalML maps are small).
 *
 * Clocking: single 27MHz domain (LiteDRAM sys = board clock, no CDC).
 */
class LiteDramAxiBridge(
  axiConfig: Axi4Config
) extends Component {
  require(axiConfig.dataWidth == 64, "LiteDRAM core port is 64-bit")
  require(axiConfig.idWidth == 4, "LiteDRAM core port idWidth is 4")

  val io = new Bundle {
    /** Standard AXI4 slave facing `DdrAdapter.extIo.ddrMaster`. */
    val axi = slave(Axi4(axiConfig))
    /** Board reset feeding the core (clock comes from the ambient domain). */
    val reset_n = in Bool()
    /** Core status (gate SoC reset on `init_done`, see `DramSoCTop`). */
    val init_done  = out Bool()
    val pll_locked = out Bool()
    /** DDR3 pads to the top level (board `.cst`). */
    val a         = out Bits(14 bits)
    val ba        = out Bits(3 bits)
    val ras_n     = out Bool()
    val cas_n     = out Bool()
    val we_n      = out Bool()
    val cs_n      = out Bool()
    val dm        = out Bits(2 bits)
    val dq        = inout(Analog(Bits(16 bits)))
    val dqs_p     = inout(Analog(Bits(2 bits)))
    val dqs_n     = inout(Analog(Bits(2 bits)))
    val clk_p     = out Bool()
    val clk_n     = out Bool()
    val cke       = out Bool()
    val odt       = out Bool()
    val ddr_reset_n = out Bool()
  }

  val core = new LiteDramCore()
  // Sanctioned BlackBox clock hookup: explicit `:=` wiring of a domain
  // clock wire dangles (clock wires live outside the signal netlist).
  // Uses the ambient domain (the SoC domain at instantiation).
  core.mapClockDomain(clock = core.io.clk27)
  core.io.reset_n := io.reset_n
  io.init_done := core.io.init_done
  io.pll_locked := core.io.pll_locked
  io.a := core.io.a
  io.ba := core.io.ba
  io.ras_n := core.io.ras_n
  io.cas_n := core.io.cas_n
  io.we_n := core.io.we_n
  io.cs_n := core.io.cs_n
  io.dm := core.io.dm
  io.dq := core.io.dq
  io.dqs_p := core.io.dqs_p
  io.dqs_n := core.io.dqs_n
  io.clk_p := core.io.clk_p
  io.clk_n := core.io.clk_n
  io.cke := core.io.cke
  io.odt := core.io.odt
  io.ddr_reset_n := core.io.reset_n_1

  // Write-address channel (single beat on the wire).
  core.io.aw_valid := io.axi.aw.valid
  io.axi.aw.ready := core.io.aw_ready
  core.io.aw_first := True
  core.io.aw_last := True
  core.io.aw_payload_addr := io.axi.aw.payload.addr(29 downto 3)
  core.io.aw_payload_burst := io.axi.aw.payload.burst
  core.io.aw_payload_len := io.axi.aw.payload.len.resized
  core.io.aw_payload_size := io.axi.aw.payload.size
  core.io.aw_payload_lock := io.axi.aw.payload.lock(0)
  core.io.aw_payload_prot := io.axi.aw.payload.prot
  core.io.aw_payload_cache := io.axi.aw.payload.cache
  core.io.aw_payload_qos := io.axi.aw.payload.qos
  core.io.aw_payload_region := io.axi.aw.payload.region
  core.io.aw_param_id := io.axi.aw.payload.id
  core.io.aw_param_dest := False
  core.io.aw_param_user := False

  // Write-data framing: first/last from a latched burst length.
  // Single outstanding write transaction (guaranteed by `DdrAdapter` bus
  // ownership), but AW and the first W beat may be accepted in the SAME
  // cycle: fresh AW fields bypass the registers in that case.
  val burstLen = Reg(UInt(8 bits)) init(0)
  val burstId = Reg(UInt(axiConfig.idWidth bits)) init(0)
  val beatCount = Reg(UInt(8 bits)) init(0)
  when(io.axi.aw.fire) {
    burstLen := io.axi.aw.payload.len.resized
    burstId := io.axi.aw.payload.id
    beatCount := 0
  }
  val freshLen = io.axi.aw.payload.len.resized
  core.io.w_valid := io.axi.w.valid
  io.axi.w.ready := core.io.w_ready
  core.io.w_first := io.axi.aw.fire || (beatCount === 0)
  core.io.w_last := beatCount === Mux(io.axi.aw.fire, freshLen, burstLen)
  core.io.w_payload_data := io.axi.w.payload.data
  core.io.w_payload_strb := io.axi.w.payload.strb
  core.io.w_param_id := Mux(io.axi.aw.fire, io.axi.aw.payload.id, burstId)
  core.io.w_param_dest := False
  core.io.w_param_user := False
  when(io.axi.w.fire) {
    when(io.axi.w.payload.last) {
      beatCount := 0
    } otherwise {
      beatCount := beatCount + 1
    }
  }

  // Write response (single response per transaction).
  io.axi.b.valid := core.io.b_valid
  core.io.b_ready := io.axi.b.ready
  io.axi.b.payload.resp := core.io.b_payload_resp
  io.axi.b.payload.id := core.io.b_param_id

  // Read-address channel (single beat on the wire).
  core.io.ar_valid := io.axi.ar.valid
  io.axi.ar.ready := core.io.ar_ready
  core.io.ar_first := True
  core.io.ar_last := True
  core.io.ar_payload_addr := io.axi.ar.payload.addr(29 downto 3)
  core.io.ar_payload_burst := io.axi.ar.payload.burst
  core.io.ar_payload_len := io.axi.ar.payload.len.resized
  core.io.ar_payload_size := io.axi.ar.payload.size
  core.io.ar_payload_lock := io.axi.ar.payload.lock(0)
  core.io.ar_payload_prot := io.axi.ar.payload.prot
  core.io.ar_payload_cache := io.axi.ar.payload.cache
  core.io.ar_payload_qos := io.axi.ar.payload.qos
  core.io.ar_payload_region := io.axi.ar.payload.region
  core.io.ar_param_id := io.axi.ar.payload.id
  core.io.ar_param_dest := False
  core.io.ar_param_user := False

  // Read data (`r_first` unused, `r_last` delimits the burst).
  io.axi.r.valid := core.io.r_valid
  core.io.r_ready := io.axi.r.ready
  io.axi.r.payload.data := core.io.r_payload_data
  io.axi.r.payload.resp := core.io.r_payload_resp
  io.axi.r.payload.last := core.io.r_last
  io.axi.r.payload.id := core.io.r_param_id
}
