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

  // Tie off unused accelerator write channels
  io.axi.aw.ready := False
  io.axi.w.ready  := False
  io.axi.b.valid  := False
  io.axi.b.payload.id := 0
  io.axi.b.payload.resp := 0

  // ------------------------------------------------------------------
  // Host write port -> AXI write transaction on external DDR
  // ------------------------------------------------------------------
  val awPending = RegInit(False)
  val wPending  = RegInit(False)
  val wrAddrReg = Reg(UInt(32 bits)) init(0)
  val wrDataReg = Reg(Bits(axiConfig.dataWidth bits)) init(0)

  when(io.wrEnable) {
    awPending := True
    wPending  := True
    wrAddrReg := io.wrAddr
    wrDataReg := io.wrData
  }

  when(extIo.ddrMaster.aw.fire) {
    awPending := False
  }

  when(extIo.ddrMaster.w.fire) {
    wPending := False
  }

  extIo.ddrMaster.aw.valid := awPending
  extIo.ddrMaster.aw.payload.addr := wrAddrReg
  extIo.ddrMaster.aw.payload.len := 0
  extIo.ddrMaster.aw.payload.size := log2Up(axiConfig.dataWidth / 8)
  extIo.ddrMaster.aw.payload.burst := B"01" // INCR
  extIo.ddrMaster.aw.payload.id := 0
  extIo.ddrMaster.aw.payload.region := 0
  extIo.ddrMaster.aw.payload.lock := 0
  extIo.ddrMaster.aw.payload.cache := 0
  extIo.ddrMaster.aw.payload.qos := 0
  extIo.ddrMaster.aw.payload.prot := 0

  extIo.ddrMaster.w.valid := wPending

  extIo.ddrMaster.w.payload.data := wrDataReg
  extIo.ddrMaster.w.payload.strb := B((1 << (axiConfig.dataWidth / 8)) - 1, (axiConfig.dataWidth / 8) bits)
  extIo.ddrMaster.w.payload.last := True

  extIo.ddrMaster.b.ready := True
}
