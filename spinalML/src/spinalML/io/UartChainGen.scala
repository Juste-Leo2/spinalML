// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config

/**
 * Generator for the complete UART hardware chain modules:
 * UartRx, UartTx, UartBridge, AxiReadMem.
 *
 * Supports named command-line arguments for full parameterization from CLI:
 *   --out <dir>            Output directory (default: rtl)
 *   --clk <Hz>             Clock frequency in Hz (default: 27000000)
 *   --baud <bps>           Baud rate in bps (default: 115200)
 *   --out-count <n>        Output stream length in bytes (default: 10)
 *   --word-width <bits>    AXI data width in bits (default: 64)
 *   --memory-words <words> BRAM depth in words (default: 4096)
 *   --img-base <hex>       Virtual base address for activations (default: 0x10000)
 *   --weight-base <hex>    Virtual base address for weights (default: 0x20000)
 */
object UartChainGen extends App {
  var targetDir: String = "rtl"
  var clkFreq: BigInt = 27000000
  var baudRate: BigInt = 115200
  var outCount: Int = 10
  var wordWidth: Int = 64
  var memoryWords: Int = 4096
  var imgBase: Int = 0x10000
  var weightBase: Int = 0x20000
  var version: Int = 0x01

  var i = 0
  while (i < args.length) {
    args(i) match {
      case "--out" | "-o"                 => targetDir = args(i + 1); i += 1
      case "--clk" | "--clk-freq"         => clkFreq = BigInt(args(i + 1)); i += 1
      case "--baud" | "--baud-rate"       => baudRate = BigInt(args(i + 1)); i += 1
      case "--out-count"                  => outCount = args(i + 1).toInt; i += 1
      case "--word-width"                 => wordWidth = args(i + 1).toInt; i += 1
      case "--memory-words" | "--bram"    => memoryWords = args(i + 1).toInt; i += 1
      case "--img-base"                   => imgBase = Integer.decode(args(i + 1)); i += 1
      case "--weight-base"                => weightBase = Integer.decode(args(i + 1)); i += 1
      case other if i == 0 && !other.startsWith("-") => targetDir = other
      case _ =>
    }
    i += 1
  }

  val cfg = Axi4Config(addressWidth = 32, dataWidth = wordWidth, idWidth = 4)

  val spinalConfig = SpinalConfig(
    targetDirectory = targetDir,
    headerWithDate = true,
    rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
  )

  println(s"[UartChainGen] Generating supplementary UART chain Verilog in '$targetDir':")
  println(s"  - Clock Frequency : $clkFreq Hz")
  println(s"  - Baud Rate       : $baudRate bps (CLK_PER_BIT = ${clkFreq / baudRate})")
  println(s"  - Out Count       : $outCount bytes")
  println(s"  - AXI Data Width  : $wordWidth bits")
  println(s"  - Memory Words    : $memoryWords (${(memoryWords * (wordWidth / 8)) / 1024} KiB BRAM)")

  spinalConfig.generateVerilog(new UartRx(clkFreq, baudRate))
  spinalConfig.generateVerilog(new UartTx(clkFreq, baudRate))
  spinalConfig.generateVerilog(new UartBridge(outCount = outCount, wordWidth = cfg.dataWidth, csrAddrWidth = 8, version = version))
  spinalConfig.generateVerilog(new AxiReadMem(cfg, memoryWords = memoryWords, imgBase = imgBase, weightBase = weightBase))
  println(s"[UartChainGen] Successfully generated UartRx.v, UartTx.v, UartBridge.v, AxiReadMem.v")
}
