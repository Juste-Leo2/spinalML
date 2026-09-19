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

    /** Host / UART loader write port. `wrStrb` is a per-byte write mask
      * (bit i enables byte i of `wrData`, LSB-first), mirroring AXI w.strb,
      * so partial-word writes never leak stale bytes from earlier commands. */
    val wrEnable = in(Bool())
    val wrAddr   = in(UInt(32 bits))
    val wrData   = in(Bits(axiConfig.dataWidth bits))
    val wrStrb   = in(Bits((axiConfig.dataWidth / 8) bits))
    val wrReady  = out(Bool())
  }
}

/**
 * Logical memory kind selecting a `MemoryAdapter` implementation.
 *
 * Decouples the *choice* of backing store from the silicon `Target`: an FPGA
 * build can run on on-chip BRAM in test and on external DRAM in production
 * without touching the Layer-1 datapath (docs/ddr_impl.md Phase 1).
 */
sealed trait MemoryKind
object MemoryKind {
  /** On-chip block RAM (FPGA BRAM / Gowin BSRAM / sim `Mem`). */
  case object OnChip extends MemoryKind
  /** Off-chip DRAM behind an AXI4 controller (DDR3/DDR4/LPDDR). */
  case object ExternalDram extends MemoryKind
  /** Compiled SRAM macros (OpenRAM Sky130 / GF180). */
  case object AsicSram extends MemoryKind
}

object MemoryAdapter {

  /**
   * Factory creating a memory adapter generator for the specified target.
   *
   * When `kind` is `None` (default, backward compatible) the implementation
   * is inferred from the target exactly as before: ASIC → `SramAsicAdapter`,
   * FPGA/Simulation → `BramAdapter`. Pass an explicit `MemoryKind` to
   * override, e.g. `(Target.FPGA(), Some(MemoryKind.ExternalDram))` for the
   * DDR bring-up while keeping BRAM for unit tests.
   */
  def factory(
    target: Target,
    kind: Option[MemoryKind] = None,
    memoryWords: Int = 4096,
    imgBase: Int = 0x10000,
    weightBase: Int = 0x20000
  ): (Axi4Config) => MemoryAdapter = { (cfg: Axi4Config) =>
    kind match {
      case Some(MemoryKind.OnChip)      => new BramAdapter(cfg, memoryWords, imgBase, weightBase)
      case Some(MemoryKind.ExternalDram) => new DdrAdapter(cfg, imgBase, weightBase)
      case Some(MemoryKind.AsicSram) =>
        val pdk = target match {
          case Target.ASIC(p, _) => p
          case _                 => spinalML.PdkFamily.Sky130
        }
        new SramAsicAdapter(cfg, memoryWords, imgBase, weightBase, pdk)
      case None => target match {
        case Target.ASIC(pdk, _) =>
          new SramAsicAdapter(cfg, memoryWords, imgBase, weightBase, pdk)
        case Target.FPGA(_, _) =>
          new BramAdapter(cfg, memoryWords, imgBase, weightBase)
        case Target.Simulation =>
          new BramAdapter(cfg, memoryWords, imgBase, weightBase)
      }
    }
  }
}

