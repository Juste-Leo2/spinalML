// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config

/**
 * Generator for the complete UART hardware chain modules:
 * UartRx, UartTx, UartBridge, AxiReadMem.
 *
 * Generates the new Verilog implementations directly from Scala sources:
 *   - UartRx.v (115200 baud @ 27 MHz)
 *   - UartTx.v (115200 baud @ 27 MHz)
 *   - UartBridge.v (L2 bridge: CSR, BRAM, stream, status, version)
 *   - AxiReadMem.v (32 KB BRAM with AXI4 read interface)
 */
object UartChainGen extends App {
  val targetDir = if (args.length > 0) args(0) else "rtl"
  val clkFreq: BigInt = 27000000
  val baudRate: BigInt = 115200
  val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val spinalConfig = SpinalConfig(
    targetDirectory = targetDir,
    headerWithDate = true,
    rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
  )

  println(s"[UartChainGen] Generating supplementary UART chain Verilog in '$targetDir'...")
  spinalConfig.generateVerilog(new UartRx(clkFreq, baudRate))
  spinalConfig.generateVerilog(new UartTx(clkFreq, baudRate))
  spinalConfig.generateVerilog(new UartBridge(outCount = 10, wordWidth = cfg.dataWidth, csrAddrWidth = 8, version = 0x01))
  spinalConfig.generateVerilog(new AxiReadMem(cfg, memoryWords = 4096, imgBase = 0x10000, weightBase = 0x20000))
  println(s"[UartChainGen] Successfully generated UartRx.v, UartTx.v, UartBridge.v, AxiReadMem.v")
}
