// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

/**
 * High-Level Residual MLP Template for SpinalML
 *
 * Demonstrates the DAG topology support of the Sequential builder: the input
 * tensor (node 0) skips over the two dense layers and is merged back through
 * an explicit Add node — the classic ResNet-style skip connection.
 *
 * Topology and shapes:
 *   [2, 4] -> Linear (4 -> 4)          node 1
 *           -> ReLU                    node 2
 *           -> Linear (4 -> 4)         node 3
 *           -> Add(node 0, node 3)     node 4   <- skip connection
 *           -> ReLU                    node 5   <- accelerator output
 *
 * The builder forks node 0 automatically: the immediate consumer reads the
 * direct path while the deferred Add consumer drains an exact-capacity FIFO.
 */
case class ResidualMLPTemplate(override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)) extends Accelerator(
  dataType = BF16(),
  inputShape = Seq(2, 4),

  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    Linear(inFeatures = 4, outFeatures = 4),
    ReLU(),
    Linear(inFeatures = 4, outFeatures = 4),
    Add(a = 0, b = 3),
    ReLU()
  ),

  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object ResidualMLPTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(ResidualMLPTemplate(axiConfig))
}
