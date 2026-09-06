// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

/**
 * Convenience DSL to build residual/skip blocks with automatic node indices.
 *
 * Node numbering convention (Sequential / ModelReplica): node 0 = input
 * tensor, layer at position p has node index p+1. An Add(a, b) references
 * those node indices directly.
 *
 * `residual(base)(branch)` appends `branch` layers plus a final Add that
 * fuses the entry node with the last branch layer:
 *   Add(base, base + branch.length)
 * which is exactly the node index of the branch tail (base = entry node,
 * each appended layer shifts the node index by +1). Pure compile-time list
 * surgery: the HW and the universal replica only see the expanded
 * Seq[LayerSpec], so no hardware change is involved.
 *
 * Example (transformer block):
 *   residual(0)(Seq(ClassicalAttention(8, 2)))
 *   // => [Attn(node1), Add(0, 1)]            (x + Attn(x), 0 = input tensor)
 *   residual(2)(Seq(Linear(8,16), ReLU(), Linear(16,8)))
 *   // => [Lin(node3), ReLU, Lin(node5), Add(2, 5)]
 */
object DagDSL {

  def residual(base: Int)(branch: Seq[LayerSpec]): Seq[LayerSpec] = {
    require(base >= 0, s"DagDSL.residual: base node $base must be >= 0 (node 0 = the input tensor)")
    require(branch.nonEmpty, "DagDSL.residual: the branch must contain at least one layer")
    branch ++ Seq(Add(a = base, b = base + branch.length))
  }
}
