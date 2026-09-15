// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

/**
 * On-chip Block RAM Memory Adapter (FPGA / Generic BRAM).
 *
 * Implements Layer 2 memory access using synchronous internal memory (Mem.readSync).
 * Supports dual virtual address regions (image vs weights) with defensive clamping
 * and 1-cycle pipelined AXI4 read-burst response machine.
 */
class BramAdapter(
  axiConfig: Axi4Config,
  val memoryWords: Int = 4096,
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000
) extends MemoryAdapter(axiConfig) {
  require(memoryWords % 2 == 0, "BramAdapter requires an even number of words (two equal regions)")
  val halfWords = memoryWords / 2
  val idxBits   = log2Up(memoryWords)
  val bytePerBeat = axiConfig.dataWidth / 8
  val shift     = log2Up(bytePerBeat)

  val mem     = Mem(Bits(axiConfig.dataWidth bits), memoryWords)
  val memData = Bits(axiConfig.dataWidth bits)

  // ------------------------------------------------------------------
  // Virtual -> physical index mapping
  // ------------------------------------------------------------------
  def mapIndex(addr: UInt): UInt = {
    val isWeight = addr >= weightBase
    // Addresses below imgBase must not underflow `addr - imgBase` into a huge
    // unsigned offset (which would silently alias the last word): clamp to the
    // first physical word. Addresses past the memory clamp to the last one.
    val offset   = Mux(isWeight, addr - weightBase,
      Mux(addr < imgBase, U(0, 32 bits), addr - imgBase))
    val raw      = (offset >> shift) + Mux(isWeight, U(halfWords, 32 bits), U(0, 32 bits))
    val clamped  = raw >= U(memoryWords - 1, raw.getWidth bits)
    Mux(clamped, U(memoryWords - 1, idxBits bits), raw(idxBits - 1 downto 0))
  }

  // Write port (Host/UART -> BRAM). The byte strobes are expanded to the
  // bit-level mask expected by Mem.write, so a partial-word host write only
  // touches the enabled bytes.
  val wrBitMask = Bits(axiConfig.dataWidth bits)
  wrBitMask := 0
  for (byte <- 0 until bytePerBeat; bit <- 0 until 8) {
    wrBitMask(byte * 8 + bit) := io.wrStrb(byte)
  }

  when(io.wrEnable) {
    mem.write(mapIndex(io.wrAddr), io.wrData, mask = wrBitMask)
  }

  // ------------------------------------------------------------------
  // AXI4 read response machine
  // ------------------------------------------------------------------
  val raddrR   = Reg(UInt(32 bits)) init 0
  val rlenR    = Reg(UInt(8 bits)) init 0
  val ridR     = Reg(UInt(axiConfig.idWidth bits)) init 0
  val rvalidR  = RegInit(False)
  val rlastR   = RegInit(False)

  val rReady = io.axi.r.ready

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

  // Registered BRAM read (1-cycle latency)
  memData := mem.readSync(mapIndex(nextRadr))

  io.axi.r.valid        := rvalidR
  io.axi.r.payload.data := memData
  io.axi.r.payload.id   := ridR
  io.axi.r.payload.last := rlastR
  io.axi.r.payload.resp := B"00"

  // ------------------------------------------------------------------
  // AXI4 write slave (accelerator write-back, e.g. DMAWriter): single
  // outstanding burst, AW accepted first (the master serializes AW/W/B),
  // every W beat committed with its byte strobes, then one B response.
  // ------------------------------------------------------------------
  val beatCountW = log2Up((1 << axiConfig.lenWidth) + 1)
  val awPending  = RegInit(False)
  val bValidR    = RegInit(False)
  val wRemaining = Reg(UInt(beatCountW bits)) init (0)
  val wAddrR     = Reg(UInt(axiConfig.addressWidth bits)) init (0)

  io.axi.aw.ready := !awPending && !bValidR
  when(io.axi.aw.valid && io.axi.aw.ready) {
    awPending  := True
    wRemaining := (io.axi.aw.payload.len +^ 1).resize(beatCountW bits)
    wAddrR     := io.axi.aw.payload.addr
  }

  io.axi.w.ready := awPending && !bValidR

  val wBeatMask = Bits(axiConfig.dataWidth bits)
  wBeatMask := 0
  for (byte <- 0 until bytePerBeat; bit <- 0 until 8) {
    wBeatMask(byte * 8 + bit) := io.axi.w.payload.strb(byte)
  }

  when(io.axi.w.valid && io.axi.w.ready) {
    mem.write(mapIndex(wAddrR), io.axi.w.payload.data, mask = wBeatMask)
    wAddrR := wAddrR + bytePerBeat
    when(wRemaining === 1) {
      awPending := False
      bValidR   := True
    } otherwise {
      wRemaining := wRemaining - 1
    }
  }

  when(bValidR && io.axi.b.ready) {
    bValidR := False
  }

  io.axi.b.valid        := bValidR
  io.axi.b.payload.id   := 0
  io.axi.b.payload.resp := B"00"
}
