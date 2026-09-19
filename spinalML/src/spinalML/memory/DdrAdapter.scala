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
  val accIdR        = Reg(UInt(axiConfig.idWidth bits)) init(0)
  val accWRemaining = Reg(UInt(beatCountW bits)) init(0)

  val accBusy   = accAwPending || accStreaming || accWaitB
  val writeBusy = hostActive || accBusy

  // Read transaction tracking
  val readInFlight = Reg(UInt(8 bits)) init(0)
  val arFire    = extIo.ddrMaster.ar.fire
  val rLastFire = extIo.ddrMaster.r.fire && extIo.ddrMaster.r.payload.last

  when(arFire && !rLastFire) {
    readInFlight := readInFlight + 1
  } elsewhen(!arFire && rLastFire) {
    readInFlight := readInFlight - 1
  }
  val readBusy = (readInFlight =/= 0)

  // ------------------------------------------------------------------
  // AXI Read: interlocked with write path to prevent RAW hazards (BUG-DDR-04)
  // ------------------------------------------------------------------
  extIo.ddrMaster.ar.valid   := io.axi.ar.valid && !writeBusy
  io.axi.ar.ready            := extIo.ddrMaster.ar.ready && !writeBusy
  extIo.ddrMaster.ar.payload := io.axi.ar.payload

  io.axi.r                   <> extIo.ddrMaster.r

  // Flow control for writes: mutually exclusive with in-flight reads
  val hostWrReady = !hostActive && !accBusy && !readBusy
  io.wrReady := hostWrReady

  when(io.wrEnable && hostWrReady) {
    hostActive := True
    hostAwDone := False
    hostWDone  := False
    wrAddrReg  := io.wrAddr
    wrDataReg  := io.wrData
    wrStrbReg  := io.wrStrb
  }

  val busHost = hostActive && !accBusy

  // Accelerator AW is accepted only while the bus is fully idle (host first, no in-flight read).
  io.axi.aw.ready := !hostActive && !accBusy && !readBusy
  when(io.axi.aw.fire) {
    accAwPending  := True
    accWRemaining := (io.axi.aw.payload.len +^ 1).resize(beatCountW bits)
    accAddrR      := io.axi.aw.payload.addr
    accIdR        := io.axi.aw.payload.id
  }

  // AW mux
  extIo.ddrMaster.aw.valid := Mux(busHost, !hostAwDone, accAwPending)
  extIo.ddrMaster.aw.payload.addr := Mux(busHost, wrAddrReg, accAddrR)
  extIo.ddrMaster.aw.payload.len := Mux(busHost, U(0, axiConfig.lenWidth bits),
    (accWRemaining -^ 1).resize(axiConfig.lenWidth bits))
  extIo.ddrMaster.aw.payload.size := log2Up(axiConfig.dataWidth / 8)
  extIo.ddrMaster.aw.payload.burst := B"01" // INCR
  extIo.ddrMaster.aw.payload.id := Mux(busHost, U(0, axiConfig.idWidth bits), accIdR)
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
