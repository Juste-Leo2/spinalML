// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.examples.Mnistw4a8

/**
 * Elaboration of the reference UART SoC top (UartSoC.v, used by
 * tests/python/test_uart_soc.py). Same configuration as UartSoCGen and the
 * reference top.v: 64-bit AXI beats, 27 MHz @ 115200, MNIST W4A8.
 */
class UartSoCTest extends AnyFunSuite {
  test("uart_soc_toplevel") {
    val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    SpinalConfig(
      headerWithDate = true,
      rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
    ).generateVerilog(new UartSoC(() => new Mnistw4a8(axiConfig = cfg), axiConfig = cfg))
  }
}
