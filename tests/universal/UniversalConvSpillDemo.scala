// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalConvSpillDemo
 *
 * Minimal spilling Conv2D model for the `spinalml test` (replica) path:
 * [6,6,2] in, K=2 -> [5,5,2] out (25 windows), KFull=8, Ks=4 (P=2) in I8,
 * DDR-backed. Same geometry as ConvReplicaSpillTest P1-5. The CLI scaffold
 * reads the DUT memory map and scales its timeout on spill; weights are
 * emitted slice-transposed (P1-4) and the oracle folds passes, so
 * `spinalml test` on this file is bit-exact — including under `--stress`
 * (ChaosDut wrapper, P1-7).
 */
case class UniversalConvSpillDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 6, 2),
  modelSpec = Seq(
    Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 4)
  ),
  axiConfig = axiConfig,
  temporal = 1,
  memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(4096))
)
