package spinalML.attention

import spinal.core._
import spinalML.nn.LayerSpec

trait AttentionCore extends LayerSpec {
  def embedDim: Int
  def numHeads: Int
}
