// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import spinal.lib.bus.amba4.axi._
import spinalML.memory.BramAdapter

/**
 * Legacy AxiReadMem component, implemented as a specialized [[BramAdapter]].
 *
 * Preserves 100% backward compatibility with existing tests, tooling, and formal verification.
 */
class AxiReadMem(
  axiConfig: Axi4Config,
  memoryWords: Int = 4096,
  imgBase: Int    = 0x10000,
  weightBase: Int = 0x20000
) extends BramAdapter(axiConfig, memoryWords, imgBase, weightBase)
