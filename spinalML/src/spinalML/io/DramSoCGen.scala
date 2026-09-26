// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.examples.Mnistw4a8

/**
 * Generation entrypoint: DRAM-backed SoC around the W4A8 MNIST accelerator,
 * same model/config as `UartSoCGen` (64-bit AXI beats, 27 MHz, 115200 baud)
 * so resource reports are directly comparable (BRAM vs LiteDRAM backing).
 * Output: DramSoCTop.v (plus the `litedram_core` BlackBox, whose Verilog
 * comes from `dram-gen`, never from SpinalHDL).
 *
 *   python cli/main.py compile spinalML/src/spinalML/io/DramSoCGen.scala --dram
 */
object DramSoCGen extends App {
  val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(new DramSoCTop(() => new Mnistw4a8(axiConfig = cfg), axiConfig = cfg))
}
