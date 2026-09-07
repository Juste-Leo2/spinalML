// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib._

/**
 * UART receiver, 1:1 port of the reference uart_rx.v (frame timing included).
 *
 * - 2-FF input resync against metastability.
 * - START bit detected, re-sampled and confirmed at the middle of the bit.
 * - DATA bits sampled at the end of the bit period (CLK_PER_BIT - 1).
 * - STOP bit validated; data is only committed when the STOP level is high.
 * - `valid` pulses for exactly one clock cycle when a byte commits.
 */
class UartRx(clkFreq: BigInt, baudRate: BigInt) extends Component {
  val clkPerBit = (clkFreq / baudRate).toInt

  val io = new Bundle {
    val rx    = in(Bool())
    val data  = out(Bits(8 bits))
    val valid = out(Bool())
  }

  val IDLE  = B"00" // wait for a falling edge
  val START = B"01" // confirm the start bit (mid-bit sample)
  val DATA  = B"10" // sample 8 data bits
  val STOP  = B"11" // validate the stop bit

  val state    = RegInit(IDLE)
  val clkCount = Reg(UInt(log2Up(clkPerBit + 1) bits)) init 0
  val bitIndex = Reg(UInt(3 bits)) init 0
  val rxData   = Reg(Bits(8 bits)) init 0

  // Resynchronization of the asynchronously sampled RX line
  val rxSync1 = RegInit(True)
  val rxSync  = RegInit(True)
  rxSync1 := io.rx
  rxSync  := rxSync1

  io.valid := False
  io.data  := rxData

  switch(state) {
    is(IDLE) {
      clkCount := 0
      bitIndex := 0
      when(!rxSync) {
        state := START
      }
    }
    is(START) {
      when(clkCount === (clkPerBit / 2)) {
        when(!rxSync) {
          // Confirmed start bit
          clkCount := 0
          state := DATA
        } otherwise {
          // False start
          state := IDLE
        }
      } otherwise {
        clkCount := clkCount + 1
      }
    }
    is(DATA) {
      when(clkCount === (clkPerBit - 1)) {
        clkCount := 0
        rxData(bitIndex) := rxSync
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
      when(clkCount === (clkPerBit - 1)) {
        clkCount := 0
        when(rxSync) {
          io.data  := rxData
          io.valid := True
        }
        state := IDLE
      } otherwise {
        clkCount := clkCount + 1
      }
    }
  }
}
