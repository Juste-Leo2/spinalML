// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalConv1DSpillDemo
 *
 * Minimal spilling Conv1D model for the `spinalml test` (replica) path:
 * [6,4] in, K=2 -> [5,2] out (5 windows), KFull=8, Ks=4 (P=2) in I8,
 * DDR-backed. Same geometry as Conv1DReplicaSpillTest P2-5. The CLI scaffold
 * reads the DUT memory map and scales its timeout on spill; weights are
 * emitted slice-transposed (P2-4) and the oracle folds passes, so
 * `spinalml test` on this file is bit-exact — including under `--stress`
 * (ChaosDut wrapper, P2-6).
 */
case class UniversalConv1DSpillDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(6, 4),
  modelSpec = Seq(
    Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2, weightLanes = 4, spillKSlice = 4)
  ),
  axiConfig = axiConfig,
  temporal = 1,
  memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(4096))
)
