// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib._

/**
 * UART loopback component used by the python golden test (test_uart.py):
 * a byte injected on io_rxIn is decoded by the receiver, re-encoded by the
 * transmitter, and both the decoded `dataOut`/`validOut` and the wire-level
 * `txOut` framing are observable from the testbench.
 */
case class UartRxTxLoopbackComp(clkFreq: BigInt = 27000000, baudRate: BigInt = 115200) extends Component {
  val io = new Bundle {
    val rxIn     = in(Bool())
    val txOut    = out(Bool())
    val dataOut  = out(Bits(8 bits))
    val validOut = out(Bool())
  }

  val rx = new UartRx(clkFreq, baudRate)
  val tx = new UartTx(clkFreq, baudRate)

  rx.io.rx := io.rxIn
  io.dataOut  := rx.io.data
  io.validOut := rx.io.valid

  tx.io.start := rx.io.valid
  tx.io.data  := rx.io.data
  io.txOut    := tx.io.tx
}

class UartRxTest extends AnyFunSuite {
  test("uart_rx_toplevel") {
    SpinalConfig(
      headerWithDate = true,
      rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
    ).generateVerilog(UartRxTxLoopbackComp(27000000, 115200))
  }
}
