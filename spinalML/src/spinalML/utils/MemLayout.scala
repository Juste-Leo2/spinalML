// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

/**
 * Single source of truth for the DDR region layout conventions shared by the
 * builder (`nn.Sequential`) and the test benches that pack weights into memory.
 */
object MemLayout {

  /** Byte size of a DDR region holding `elements` values of `bits` width.
    * The ceil is applied on the WHOLE region so sub-byte element types
    * (e.g. I4 nibbles, two-per-byte) don't floor to zero bytes and make the
    * next region alias this one's start. For byte-multiple widths the ceil
    * is exact, so the same formula serves I8/I16/BF16/... Callers must
    * beat-align region starts afterwards (see [[alignToBeat]]). */
  def regionBytes(elements: Int, bits: Int): Int = (elements * bits + 7) / 8

  /** Round a byte offset up to the AXI beat size (`beatBytes = dataWidth/8`):
    * DDR controllers and memory models serve bursts from beat-aligned
    * addresses, so an unaligned region start would silently read the tail of
    * the previous region's word. */
  def alignToBeat(offset: Int, beatBytes: Int): Int =
    (offset + beatBytes - 1) / beatBytes * beatBytes
}
