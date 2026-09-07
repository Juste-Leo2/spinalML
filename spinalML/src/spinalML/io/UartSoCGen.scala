// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.examples.Mnistw4a8

/**
 * Generation entrypoint: UART SoC around the W4A8 MNIST accelerator,
 * reference top.v-equivalent configuration (64-bit AXI beats, 27 MHz,
 * 115200 baud). Output: UartSoC.v
 *
 *   python cli/main.py compile spinalML/src/spinalML/io/UartSoCGen.scala
 */
object UartSoCGen extends App {
  val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(new UartSoC(() => new Mnistw4a8(axiConfig = cfg), axiConfig = cfg))
}
