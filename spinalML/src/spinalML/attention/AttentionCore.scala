// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.attention

import spinal.core._
import spinalML.nn.LayerSpec

trait AttentionCore extends LayerSpec {
  def embedDim: Int
  def numHeads: Int
}
