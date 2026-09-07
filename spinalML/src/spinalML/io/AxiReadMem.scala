// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

/**
 * Internal BRAM (memoryWords words of axiConfig.dataWidth bits) that serves
 * the accelerator AXI-master reads with the exact response timing of the
 * reference top.v BRAM block, and receives UART bridge writes.
 *
 * Virtual to physical mapping (two equal-size regions, mirror of top.v):
 *   index = (addr >= weightBase) ? ((addr - weightBase) >> shift) + halfWords
 *                                : ((addr - imgBase) >> shift)
 * Out-of-range addresses are clamped to the last word (defensive; reference
 * behaviour would read garbage out of the RAM).
 *
 * Response protocol (AXI read): ar.ready = !r.valid, data registered with
 * 1-cycle latency (mirrors `ram[idx]` -> `axi_rdata_reg`), resp = 0,
 * `id` preserved, `len` in AXI (beats = len + 1), addresses walk by
 * bytePerBeat (64-bit beat -> +8).
 */
class AxiReadMem(
  val axiConfig: Axi4Config,
  val memoryWords: Int = 4096,
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000
) extends Component {
  require(memoryWords % 2 == 0, "AxiReadMem requires an even number of words (two equal regions)")
  val halfWords = memoryWords / 2
  val idxBits   = log2Up(memoryWords)
  val bytePerBeat = axiConfig.dataWidth / 8
  val shift     = log2Up(bytePerBeat)

  val io = new Bundle {
    val axi      = slave(Axi4(axiConfig))
    val wrEnable = in(Bool())
    val wrAddr   = in(UInt(32 bits)) // virtual address, mapped internally
    val wrData   = in(Bits(axiConfig.dataWidth bits))
  }

  val mem     = Mem(Bits(axiConfig.dataWidth bits), memoryWords)
  val memData = Bits(axiConfig.dataWidth bits)

  // ------------------------------------------------------------------
  // Virtual -> physical index (shared formula with the write port)
  // ------------------------------------------------------------------
  def mapIndex(addr: UInt): UInt = {
    val isWeight = addr >= weightBase
    val offset   = Mux(isWeight, addr - weightBase, addr - imgBase)
    val raw      = (offset >> shift) + Mux(isWeight, U(halfWords, 32 bits), U(0, 32 bits))
    val clamped  = raw >= U(memoryWords - 1, raw.getWidth bits)
    Mux(clamped, U(memoryWords - 1, idxBits bits), raw(idxBits - 1 downto 0))
  }

  // Write port (UART bridge -> BRAM)
  when(io.wrEnable) {
    mem.write(mapIndex(io.wrAddr), io.wrData)
  }

  // ------------------------------------------------------------------
  // AXI read response machine (1:1 timing mirror of top.v)
  // ------------------------------------------------------------------
  val raddrR   = Reg(UInt(32 bits)) init 0
  val rlenR    = Reg(UInt(8 bits)) init 0
  val ridR     = Reg(UInt(axiConfig.idWidth bits)) init 0
  val rvalidR  = RegInit(False)
  val rlastR   = RegInit(False)

  val rReady = io.axi.r.ready

  // Combinational next read address, exact mirror of top.v `next_raddr`
  val nextRadr = Mux(io.axi.ar.valid && io.axi.ar.ready,
    io.axi.ar.payload.addr,
    Mux(rvalidR && rReady && (rlenR > 0), raddrR + bytePerBeat, raddrR))

  io.axi.ar.ready := !rvalidR

  when(io.axi.ar.valid && io.axi.ar.ready) {
    raddrR  := io.axi.ar.payload.addr
    rlenR   := io.axi.ar.payload.len
    ridR    := io.axi.ar.payload.id
    rvalidR := True
    rlastR  := io.axi.ar.payload.len === 0
  } elsewhen(rvalidR && rReady) {
    when(rlenR === 0) {
      rvalidR := False
      rlastR  := False
    } otherwise {
      rlenR   := rlenR - 1
      raddrR  := raddrR + bytePerBeat
      rlastR  := rlenR === 1
    }
  }

  // Registered BRAM read (1-cycle latency, always refreshed as in top.v)
  memData := mem.readSync(mapIndex(nextRadr))

  io.axi.r.valid        := rvalidR
  io.axi.r.payload.data := memData
  io.axi.r.payload.id   := ridR
  io.axi.r.payload.last := rlastR
  io.axi.r.payload.resp := B"00"

  // Write channels are not used by the accelerator (read-only DDR):
  io.axi.aw.ready := False
  io.axi.w.ready := False
  io.axi.b.valid := False
  io.axi.b.payload.id := 0
  io.axi.b.payload.resp := 0
}
