// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

/**
 * Off-chip External DRAM Memory Adapter (DDR3 / DDR4 / DDR5 / LPDDR).
 *
 * Connects the Neural Network Accelerator and Host Write Port to an external
 * AXI4-compliant DRAM controller (e.g. Gowin GW2A DDR3 IP on Tang Primer 20K).
 */
class DdrAdapter(
  axiConfig: Axi4Config,
  val imgBase: Int    = 0x10000,
  val weightBase: Int = 0x20000
) extends MemoryAdapter(axiConfig) {

  val extIo = new Bundle {
    /** Master AXI4 interface to the external DDR controller slave port. */
    val ddrMaster = master(Axi4(axiConfig))
  }

  // ------------------------------------------------------------------
  // AXI Read: pass-through from Accelerator master to external DDR
  // ------------------------------------------------------------------
  extIo.ddrMaster.ar <> io.axi.ar
  io.axi.r           <> extIo.ddrMaster.r

  // ------------------------------------------------------------------
  // Shared AXI4 write path: the host single-beat write (wrEnable) and the
  // accelerator burst (DMAWriter write-back) are mutually exclusive on the
  // external DDR master. The host has priority once its write is latched;
  // an accelerator burst owns the bus from its accepted AW until its B
  // response.
  // ------------------------------------------------------------------
  val beatCountW = log2Up((1 << axiConfig.lenWidth) + 1)

  // Host transaction state
  val hostActive = RegInit(False)
  val hostAwDone = RegInit(False)
  val hostWDone  = RegInit(False)
  val wrAddrReg  = Reg(UInt(32 bits)) init(0)
  val wrDataReg  = Reg(Bits(axiConfig.dataWidth bits)) init(0)
  val wrStrbReg  = Reg(Bits(axiConfig.dataWidth / 8 bits)) init(0)

  // Accelerator transaction state
  val accAwPending  = RegInit(False)
  val accStreaming  = RegInit(False)
  val accWaitB      = RegInit(False)
  val accAddrR      = Reg(UInt(axiConfig.addressWidth bits)) init(0)
  val accWRemaining = Reg(UInt(beatCountW bits)) init(0)

  when(io.wrEnable) {
    hostActive := True
    hostAwDone := False
    hostWDone  := False
    wrAddrReg  := io.wrAddr
    wrDataReg  := io.wrData
    wrStrbReg  := io.wrStrb
  }

  val accBusy = accAwPending || accStreaming || accWaitB
  val busHost = hostActive && !accBusy

  // Accelerator AW is accepted only while the bus is fully idle (host first).
  io.axi.aw.ready := !hostActive && !accBusy
  when(io.axi.aw.fire) {
    accAwPending  := True
    accWRemaining := (io.axi.aw.payload.len +^ 1).resize(beatCountW bits)
    accAddrR      := io.axi.aw.payload.addr
  }

  // AW mux
  extIo.ddrMaster.aw.valid := Mux(busHost, !hostAwDone, accAwPending)
  extIo.ddrMaster.aw.payload.addr := Mux(busHost, wrAddrReg, accAddrR)
  extIo.ddrMaster.aw.payload.len := Mux(busHost, U(0, axiConfig.lenWidth bits),
    (accWRemaining -^ 1).resize(axiConfig.lenWidth bits))
  extIo.ddrMaster.aw.payload.size := log2Up(axiConfig.dataWidth / 8)
  extIo.ddrMaster.aw.payload.burst := B"01" // INCR
  extIo.ddrMaster.aw.payload.id := 0
  extIo.ddrMaster.aw.payload.region := 0
  extIo.ddrMaster.aw.payload.lock := 0
  extIo.ddrMaster.aw.payload.cache := 0
  extIo.ddrMaster.aw.payload.qos := 0
  extIo.ddrMaster.aw.payload.prot := 0

  when(extIo.ddrMaster.aw.fire) {
    when(busHost) {
      hostAwDone := True
    } otherwise {
      accAwPending := False
      accStreaming := True
    }
  }

  // W mux: accelerator beats are forwarded combinationally with backpressure,
  // their `last` is generated from the remaining-beat counter.
  extIo.ddrMaster.w.valid := Mux(busHost, !hostWDone, accStreaming && io.axi.w.valid)
  extIo.ddrMaster.w.payload.data := Mux(busHost, wrDataReg, io.axi.w.payload.data)
  extIo.ddrMaster.w.payload.strb := Mux(busHost, wrStrbReg, io.axi.w.payload.strb)
  extIo.ddrMaster.w.payload.last := Mux(busHost, True, accWRemaining === 1)
  io.axi.w.ready := !busHost && accStreaming && extIo.ddrMaster.w.ready

  when(extIo.ddrMaster.w.valid && extIo.ddrMaster.w.ready) {
    when(busHost) {
      hostWDone := True
    } otherwise {
      accWRemaining := accWRemaining - 1
      when(accWRemaining === 1) {
        accStreaming := False
        accWaitB     := True
      }
    }
  }

  // B: the host response is auto-acknowledged, the accelerator burst response
  // is forwarded (one response per transaction, selected by `accWaitB`).
  extIo.ddrMaster.b.ready := Mux(accWaitB, io.axi.b.ready, True)
  io.axi.b.valid   := extIo.ddrMaster.b.valid && accWaitB
  io.axi.b.payload := extIo.ddrMaster.b.payload

  when(extIo.ddrMaster.b.valid && extIo.ddrMaster.b.ready) {
    when(accWaitB) {
      accWaitB := False
    } otherwise {
      hostActive := False
    }
  }
}
