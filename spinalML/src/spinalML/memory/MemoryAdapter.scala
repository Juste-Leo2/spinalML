// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.Target

/**
 * Universal Memory Adapter (Layer 2 abstraction).
 *
 * Decouples the Neural Network Accelerator (AXI master) and Host Loader (write port)
 * from the underlying physical storage technology:
 * - BramAdapter: On-chip block RAM (FPGA BRAM / Gowin BSRAM / Xilinx Block RAM).
 * - SramAsicAdapter: On-chip compiled SRAM macros (OpenRAM Sky130 / GF180).
 * - DdrAdapter: Off-chip DRAM (DDR3 / DDR4 / DDR5 / LPDDR controller).
 */
abstract class MemoryAdapter(val axiConfig: Axi4Config) extends Component {
  val io = new Bundle {
    /** AXI4 slave read interface connected to the Neural Network Accelerator master. */
    val axi = slave(Axi4(axiConfig))

    /** Host / UART loader write port. */
    val wrEnable = in(Bool())
    val wrAddr   = in(UInt(32 bits))
    val wrData   = in(Bits(axiConfig.dataWidth bits))
  }
}

object MemoryAdapter {

  /** Factory creating a memory adapter generator corresponding to the specified Target. */
  def factory(
    target: Target,
    memoryWords: Int = 4096,
    imgBase: Int = 0x10000,
    weightBase: Int = 0x20000
  ): (Axi4Config) => MemoryAdapter = { (cfg: Axi4Config) =>
    target match {
      case Target.ASIC(pdk, _) =>
        new SramAsicAdapter(cfg, memoryWords, imgBase, weightBase, pdk)
      case Target.FPGA(_, _) =>
        new BramAdapter(cfg, memoryWords, imgBase, weightBase)
      case Target.Simulation =>
        new BramAdapter(cfg, memoryWords, imgBase, weightBase)
    }
  }
}

