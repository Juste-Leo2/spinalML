// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.{PdkFamily, Target}

/**
 * On-chip Compiled SRAM Memory Adapter for ASIC flows (OpenRAM / Sky130 / GF180).
 *
 * Provides Layer 2 memory interfacing with compiled ASIC SRAM macros.
 * Manages AXI4 burst transactions, virtual-to-physical address translation,
 * and OpenRAM-compatible control signals (CSB, WEB, ADDR, DIN, DOUT).
 */
class SramAsicAdapter(
  axiConfig: Axi4Config,
  val memoryWords: Int = 4096,
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000,
  val pdk: PdkFamily  = PdkFamily.Sky130
) extends MemoryAdapter(axiConfig) {
  require(memoryWords % 2 == 0, "SramAsicAdapter requires an even number of words")
  val halfWords = memoryWords / 2
  val idxBits   = log2Up(memoryWords)
  val bytePerBeat = axiConfig.dataWidth / 8
  val shift     = log2Up(bytePerBeat)

  // Behavioral SRAM core (emulates compiled macro behavior during simulation & behavioral synthesis)
  val sramCore = Mem(Bits(axiConfig.dataWidth bits), memoryWords)
  val sramDout = Bits(axiConfig.dataWidth bits)

  // Virtual -> physical index mapping
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

  // Host write byte strobes are passed straight to Mem.write as the byte-level
  // mask: the mask width drives the Mem symbol width, so one mask bit per
  // byte yields byte-wide memory symbols (which pack into full-width SRAM
  // macros) instead of one single-bit RAM per data bit.

  // AXI4 read response machine
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

  // Synchronous read (1-cycle macro access latency)
  sramDout := sramCore.readSync(mapIndex(nextRadr))

  io.axi.r.valid        := rvalidR
  io.axi.r.payload.data := sramDout
  io.axi.r.payload.id   := ridR
  io.axi.r.payload.last := rlastR
  io.axi.r.payload.resp := B"00"

  // AXI4 write slave (accelerator write-back, e.g. DMAWriter): single
  // outstanding burst, AW accepted first, byte-strobed W beats, one B.
  val beatCountW = log2Up((1 << axiConfig.lenWidth) + 1)
  val awPending  = RegInit(False)
  val bValidR    = RegInit(False)
  val wRemaining = Reg(UInt(beatCountW bits)) init (0)
  val wAddrR     = Reg(UInt(axiConfig.addressWidth bits)) init (0)
  val awIdR      = Reg(UInt(axiConfig.idWidth bits)) init (0)

  io.axi.aw.ready := !awPending && !bValidR
  when(io.axi.aw.valid && io.axi.aw.ready) {
    awPending  := True
    wRemaining := (io.axi.aw.payload.len +^ 1).resize(beatCountW bits)
    wAddrR     := io.axi.aw.payload.addr
    awIdR      := io.axi.aw.payload.id
  }

  io.axi.w.ready := awPending && !bValidR && !io.wrEnable
  io.wrReady := True

  // Single physical write port shared by the host loader and the accelerator
  // write-back (host priority; the accelerator beat is stalled, never
  // dropped). A 1-read + 1-write memory maps to a compiled 1R1W SRAM macro;
  // a second write port would prevent macro mapping and expand the memory to
  // flip-flops.
  val wrFire    = io.wrEnable || (io.axi.w.valid && io.axi.w.ready)
  val wrAddrSel = Mux(io.wrEnable, io.wrAddr, wAddrR)
  val wrDataSel = Mux(io.wrEnable, io.wrData, io.axi.w.payload.data)
  val wrMaskSel = Mux(io.wrEnable, io.wrStrb, io.axi.w.payload.strb)

  when(wrFire) {
    sramCore.write(mapIndex(wrAddrSel), wrDataSel, mask = wrMaskSel)
  }

  when(io.axi.w.valid && io.axi.w.ready) {
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
  io.axi.b.payload.id   := awIdR
  io.axi.b.payload.resp := B"00"
}
