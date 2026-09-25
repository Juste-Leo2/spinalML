// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

/**
 * Frozen AXI-Lite CSR address map of `Accelerator`.
 *
 * Single source of truth for the control register addresses so host drivers,
 * test benches and the RTL can never drift apart. Values are identical to the
 * historical literals previously scattered in `Accelerator` — this object only
 * names them.
 *
 * Reserved (not yet wired): `SpillBase`, `OutStride`, `MemStatus`.
 */
object CsrMap {
  /** 0x00: Control — write 1 to trigger one inference. */
  val Start: Int = 0x00
  /** 0x04: Status — bit0 done, bit1 busy, bit2 RUN state. */
  val Status: Int = 0x04
  /** 0x08: Image base address (DDR address of the input tensor). */
  val ImgBase: Int = 0x08
  /** 0x0C: Weights base address (DDR address of the weight/bias region). */
  val WeightBase: Int = 0x0C
  /** 0x10: MODE — bit0 WEIGHT_RESIDENT, bit1 PREFETCH_EN. */
  val Mode: Int = 0x10
  /** 0x14: RELOAD — any write pulses a one-shot weight/bias refetch. */
  val Reload: Int = 0x14
  /** 0x18: TILE_CNT (RO) — completed output frames since reset. */
  val TileCnt: Int = 0x18
  /** 0x1C: RUN — bit0 auto-advance for continuous streaming. */
  val Run: Int = 0x1C
  /** 0x20: Output base address (for DMAWriter write-back). */
  val OutAddr: Int = 0x20
  /** 0x24: Output control — bit0 writeToDdr enable. */
  val OutCtrl: Int = 0x24
  /** 0x28: DMA write status — bit0 busy, bit1 done. */
  val DmaStatus: Int = 0x28
  /** 0x30: Runtime dequantization scale (Cast layers with runtimeScale). */
  val DequantScale: Int = 0x30
  /** 0x34: Spill region base address (accumulator spill). */
  val SpillBase: Int = 0x34

  // Reserved for the stride/status plane.
  /** 0x38 (RESERVED): output stride between consecutive frames. */
  val OutStride: Int = 0x38
  /** 0x3C (RESERVED): memory status (fit/fence state). */
  val MemStatus: Int = 0x3C

  /** All currently wired addresses (excludes reserved). */
  val wired: Set[Int] = Set(Start, Status, ImgBase, WeightBase, Mode, Reload,
    TileCnt, Run, OutAddr, OutCtrl, DmaStatus, DequantScale, SpillBase)

  /** All reserved addresses (must not collide with wired). */
  val reserved: Set[Int] = Set(OutStride, MemStatus)
}
