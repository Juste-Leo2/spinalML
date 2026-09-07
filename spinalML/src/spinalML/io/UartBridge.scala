// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._

/**
 * L2 command bridge, port of the reference top.v UART state machine.
 *
 * One byte at a time, all multi-byte fields received MSB-first (first
 * received byte is the most significant — see docs/uart_bridge.md for the
 * proven-by-RTL rationale; payload data words are assembled LSB-first).
 *
 * Commands (opcode on the first byte):
 *   'C' 0x43: CSR write  (4B addr + 4B val, AXI-lite write, strb = 1111)
 *   'W' 0x57: BRAM write (4B addr + 4B len + len bytes of payload)
 *   'R' 0x52: read outCount bytes from the accelerator output stream
 *   'S' 0x53: send the 8-bit status byte
 *   'V' 0x50: send the protocol version byte (additive)
 *
 * Notes versus reference code:
 *  - dead reference state S_C_EXEC_W is not ported (C writes go straight to
 *    S_C_EXEC_AW in the Verilog and deassert both valids there).
 *  - the reference address/len increment via the delayed w_bram_en flag is
 *    ported 1:1 through the `wrEnableR` register (write at flag+1 with the
 *    pre-increment address, increment at the same edge).
 */
class UartBridge(
  val outCount: Int       = 10,
  val wordWidth: Int      = 64,
  val csrAddrWidth: Int   = 8,
  val version: Int        = 0x01
) extends Component {
  val wordBytes = wordWidth / 8

  val IDLE       = B"0000"
  val C_ADDR     = B"0001"
  val C_VAL      = B"0010"
  val C_EXEC_AW  = B"0011"
  val C_EXEC_B   = B"0100"
  val W_ADDR     = B"0101"
  val W_LEN      = B"0110"
  val W_DATA     = B"0111"
  val R_WAIT     = B"1000"
  val R_SEND     = B"1001"
  val S_SEND     = B"1010"

  val io = new Bundle {
    // UART physical bytes (driven by UartRx / UartTx)
    val rx = slave(Stream(Bits(8 bits)))
    val tx = master(Stream(Bits(8 bits)))

    // Accelerator control bus (write path only)
    val csr = master(AxiLite4(AxiLite4Config(addressWidth = csrAddrWidth, dataWidth = 32)))

    // BRAM write port (virtual address, mapped inside AxiReadMem)
    val wrEnable = out(Bool())
    val wrAddr   = out(UInt(32 bits))
    val wrData   = out(Bits(wordWidth bits))

    // Accelerator output stream (one FP8 byte per logit)
    val outStream = slave(Stream(Bits(8 bits)))

    // Status sources (live signals of the SoC)
    val statusArValid = in(Bool())
    val statusRValid  = in(Bool())
    val accBusy       = in(Bool())
    val accDone       = in(Bool())
  }

  // ------------------------------------------------------------------
  // Registers (1:1 with top.v)
  // ------------------------------------------------------------------
  val state   = RegInit(IDLE)
  val byteCnt = Reg(UInt(3 bits)) init 0
  val addrReg = Reg(UInt(32 bits)) init 0
  val valReg  = Reg(UInt(32 bits)) init 0
  val lenReg  = Reg(UInt(32 bits)) init 0
  val logitCnt = Reg(UInt(log2Up(outCount) bits)) init 0

  val txDataR  = Reg(Bits(8 bits)) init 0
  val txStartR = RegInit(False)

  val csrAwAddrR = Reg(UInt(csrAddrWidth bits)) init 0
  val csrWDataR  = Reg(UInt(32 bits)) init 0
  val csrAwValidR = RegInit(False)
  val csrWValidR  = RegInit(False)
  val bReadyR     = RegInit(False)

  val wrEnableR = RegInit(False)
  val wrBytes   = Vec(Reg(Bits(8 bits)) init 0, wordBytes)

  // Helper: {rxByte, acc[31:8]} — LSB-first accumulation (shifts right by 8)
  def shiftIn(rxByte: Bits, acc: UInt): UInt = (rxByte ## acc(31 downto 8)).asUInt

  io.wrEnable := wrEnableR
  io.wrAddr   := addrReg
  io.wrData   := wrBytes.asBits
  io.rx.ready := True
  io.tx.valid := txStartR
  io.tx.payload := txDataR
  io.csr.aw.valid := csrAwValidR
  io.csr.aw.payload.addr := csrAwAddrR
  io.csr.aw.payload.prot := B"000"
  io.csr.w.valid := csrWValidR
  io.csr.w.payload.data := csrWDataR.asBits
  io.csr.w.payload.strb := B"1111"
  io.csr.ar.valid := False
  io.csr.ar.payload.assignDontCare()
  io.csr.r.ready := False
  io.csr.b.ready := bReadyR
  io.outStream.ready := (state === R_WAIT) && io.tx.ready

  val statusByte = io.statusArValid ## io.statusRValid ## csrAwValidR ## csrWValidR ##
                   io.outStream.valid ## io.accDone ## io.accBusy ## B"1"

  // ------------------------------------------------------------------
  // Defaults (w_bram_en-style pulse, tx_start-style pulse)
  // ------------------------------------------------------------------
  wrEnableR := False
  txStartR  := False

  // Delayed virtual address walk, exact mirror of the Verilog
  // `if (w_bram_en) addr_reg <= addr_reg + 8;`
  when(wrEnableR) {
    addrReg := addrReg + wordBytes
  }

  switch(state) {
    is(IDLE) {
      byteCnt := 0
      when(io.rx.valid) {
        switch(io.rx.payload) {
          is(0x43) { state := C_ADDR }
          is(0x57) { state := W_ADDR }
          is(0x52) { logitCnt := 0; state := R_WAIT }
          is(0x53) {
            txDataR := statusByte
            txStartR := True
            state := S_SEND
          }
          is(0x56) {  // 'V' (ASCII, same family as C/W/R/S)
            txDataR := U(version, 8 bits).asBits
            txStartR := True
            state := S_SEND
          }
        }
      }
    }

    is(C_ADDR) {
      when(io.rx.valid) {
        addrReg := shiftIn(io.rx.payload, addrReg)
        when(byteCnt === 3) {
          byteCnt := 0
          state := C_VAL
        } otherwise {
          byteCnt := byteCnt + 1
        }
      }
    }

    is(C_VAL) {
      when(io.rx.valid) {
        valReg := shiftIn(io.rx.payload, valReg)
        when(byteCnt === 3) {
          csrAwAddrR := addrReg(7 downto 0)
          csrWDataR  := shiftIn(io.rx.payload, valReg)
          csrAwValidR := True
          csrWValidR  := True
          byteCnt := 0
          state := C_EXEC_AW
        } otherwise {
          byteCnt := byteCnt + 1
        }
      }
    }

    is(C_EXEC_AW) {
      when(io.csr.aw.ready && csrAwValidR) { csrAwValidR := False }
      when(io.csr.w.ready && csrWValidR) { csrWValidR := False }
      when((io.csr.aw.ready || !csrAwValidR) && (io.csr.w.ready || !csrWValidR)) {
        state := C_EXEC_B
      }
    }

    is(C_EXEC_B) {
      bReadyR := True
      when(io.csr.b.valid && bReadyR) {
        bReadyR := False
        state := IDLE
      }
    }

    is(W_ADDR) {
      when(io.rx.valid) {
        addrReg := shiftIn(io.rx.payload, addrReg)
        when(byteCnt === 3) {
          byteCnt := 0
          state := W_LEN
        } otherwise {
          byteCnt := byteCnt + 1
        }
      }
    }

    is(W_LEN) {
      when(io.rx.valid) {
        val nextLen = shiftIn(io.rx.payload, lenReg)
        lenReg := nextLen
        when(byteCnt === 3) {
          byteCnt := 0
          when(nextLen === 0) {
            state := IDLE
          } otherwise {
            state := W_DATA
          }
        } otherwise {
          byteCnt := byteCnt + 1
        }
      }
    }

    is(W_DATA) {
      when(io.rx.valid) {
        wrBytes(byteCnt.resize(log2Up(wordBytes) max 1)) := io.rx.payload
        lenReg := lenReg - 1
        when(byteCnt === (wordBytes - 1) || lenReg === 1) {
          wrEnableR := True
          byteCnt := 0
          when(lenReg === 1) {
            state := IDLE
          }
        } otherwise {
          byteCnt := byteCnt + 1
        }
      }
    }

    is(R_WAIT) {
      when(io.outStream.valid && io.tx.ready) {
        txDataR  := io.outStream.payload
        txStartR := True
        state := R_SEND
      }
    }

    is(R_SEND) {
      when(io.tx.ready && !txStartR) {
        when(logitCnt === (outCount - 1)) {
          state := IDLE
        } otherwise {
          logitCnt := logitCnt + 1
          state := R_WAIT
        }
      }
    }

    is(S_SEND) {
      when(io.tx.ready && !txStartR) {
        state := IDLE
      }
    }
  }
}
