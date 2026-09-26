// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package tests.universal

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._
import spinalML.memory.MemoryKind

/**
 * UniversalScaleDemo
 *
 * Scale proof for the DRAM path (`spinalml test --stress` + `build --dram`):
 * a single I8 Linear K=1024, N=64, Ks=256 (P=4 passes) holding 64KiB of
 * weights — twice the on-chip BRAM budget (4096 words = 32KiB) — so the
 * model CANNOT run resident: weights must stream from external DRAM and
 * M*N partials spill between passes.
 *
 * - `memory.kind = ExternalDram` with the 128MiB Tang Primer 20K capacity:
 *   elaborates the DRAM descriptor and fit-checks against it.
 * - `spillBase = 0x40000` sits past the 64KiB weight region
 *   (weightBase 0x20000 + 0x10000), no overlap.
 * - Weights are emitted slice-transposed, the oracle folds passes
 *   (same R1/R2 machinery as UniversalSpillDemo), timeout scales on spill.
 */
case class UniversalScaleDemo(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),
  inputShape = Seq(1, 1024),
  modelSpec = Seq(
    Linear(inFeatures = 1024, outFeatures = 64, weightLanes = 8, spillKSlice = 256)
  ),
  axiConfig = axiConfig,
  temporal = 1,
  memory = MemorySpec(
    kind = MemoryKind.ExternalDram,
    spillBase = Some(0x40000L),
    capacityBytes = Some(134217728L)
  )
)
