// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * UniversalSpillDemo
 *
 * Minimal spilling model for the `spinalml test` (replica) path: a single
 * Linear K=8, N=4, Ks=4 (P=2) in I8, DDR-backed. The CLI scaffold reads the
 * DUT memory map (img/weight bases) and scales its timeout on spill;
 * weights are emitted slice-transposed by WeightMemoryLayout (R1) and the
 * oracle folds passes (R2), so `spinalml test` on this file is bit-exact.
 */
case class UniversalSpillDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(1, 8),
  modelSpec = Seq(
    Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)
  ),
  axiConfig = axiConfig,
  temporal = 1,
  memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(4096))
)
