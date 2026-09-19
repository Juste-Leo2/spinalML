// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinalML.memory.MemoryKind

/**
 * Logical memory descriptor for the high-level API (Phase-2 DDR plumbing,
 * docs/ddr_impl.md).
 *
 * Layer-2 logical view (region bases + optional capacity), deliberately
 * separated from the Layer-3 physical board data (boards JSON memory
 * block, Phase 3). The datapath (`Sequential`) only ever sees byte offsets
 * from these bases; the `MemoryAdapter` implementation behind them is
 * selected via `MemoryKind` (see `MemoryAdapter.factory`).
 *
 * `capacityBytes = None` (default, legacy) disables the fit check so every
 * existing model elaborates exactly as before. Set it to enable
 * `reportFit`, e.g. from the board profile, to fail fast at elaboration
 * with an actionable message instead of silently overflowing on-chip RAM.
 */
case class MemorySpec(
  kind: MemoryKind = MemoryKind.OnChip,
  imgBase: Long = 0x10000L,
  weightBase: Long = 0x20000L,
  /** Spill region base (accumulator spill, Phase 4/5). None = no spill. */
  spillBase: Option[Long] = None,
  /** Total usable bytes behind this descriptor. None = no fit check. */
  capacityBytes: Option[Long] = None
) {
  require(imgBase >= 0 && weightBase >= 0, s"MemorySpec bases must be non-negative (img=$imgBase weight=$weightBase)")
  spillBase.foreach(b => require(b >= 0, s"MemorySpec spillBase must be non-negative (got $b)"))
  capacityBytes.foreach(c => require(c > 0, s"MemorySpec capacityBytes must be > 0 (got $c)"))

  /**
   * Elaboration-time footprint check: image + weight/bias regions + one
   * output frame must fit in `capacityBytes` when declared.
   *
   * All sizes are exact region bytes (`MemLayout` conventions: whole-region
   * ceil + beat alignment, as computed by `Sequential.totalWeightBytes`).
   */
  def reportFit(imageBytes: Long, weightBytes: Long, outBytes: Long): Unit =
    capacityBytes.foreach { cap =>
      val total = imageBytes + weightBytes + outBytes
      require(total <= cap,
        s"MemorySpec: model footprint ${total}B (image ${imageBytes}B + weights ${weightBytes}B + out ${outBytes}B) " +
        s"exceeds capacity ${cap}B — enable activation tiling (tileHeight), accumulator spill, " +
        s"layer folding, or quantized weights (see docs/wave6_ddr_scaling_plan.md)")
    }
}

object MemorySpec {
  /** Legacy descriptor: on-chip BRAM, historical bases, no fit check. */
  def default: MemorySpec = MemorySpec()
}
