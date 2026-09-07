// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._

/**
 * UART transmitter, 1:1 port of the reference uart_tx.v (frame timing included).
 *
 * - `ready` is high while idle; a `start` pulse latches `data` and sends it.
 * - Frame: 1 start bit (low), 8 data bits LSB-first, 1 stop bit (high).
 * - `ready` goes low during the whole frame and returns high in IDLE.
 */
class UartTx(clkFreq: BigInt, baudRate: BigInt) extends Component {
  val clkPerBit = (clkFreq / baudRate).toInt

  val io = new Bundle {
    val start = in(Bool())
    val data  = in(Bits(8 bits))
    val tx    = out(Bool())
    val ready = out(Bool())
  }

  val IDLE  = B"00"
  val START = B"01"
  val DATA  = B"10"
  val STOP  = B"11"

  val state    = RegInit(IDLE)
  val clkCount = Reg(UInt(log2Up(clkPerBit + 1) bits)) init 0
  val bitIndex = Reg(UInt(3 bits)) init 0
  val txData   = Reg(Bits(8 bits)) init 0
  val tx       = RegInit(True)
  val ready    = RegInit(True)

  switch(state) {
    is(IDLE) {
      tx      := True
      ready   := True
      clkCount := 0
      bitIndex := 0
      when(io.start) {
        txData := io.data
        ready  := False
        state  := START
      }
    }
    is(START) {
      tx := False
      when(clkCount === (clkPerBit - 1)) {
        clkCount := 0
        state := DATA
      } otherwise {
        clkCount := clkCount + 1
      }
    }
    is(DATA) {
      tx := txData(bitIndex)
      when(clkCount === (clkPerBit - 1)) {
        clkCount := 0
        when(bitIndex === 7) {
          state := STOP
        } otherwise {
          bitIndex := bitIndex + 1
        }
      } otherwise {
        clkCount := clkCount + 1
      }
    }
    is(STOP) {
      tx := True
      when(clkCount === (clkPerBit - 1)) {
        clkCount := 0
        state := IDLE
      } otherwise {
        clkCount := clkCount + 1
      }
    }
  }

  io.tx    := tx
  io.ready := ready
}
