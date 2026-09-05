// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.accelerator

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axis._
import spinalML.interfaces._
import spinalML.ops._
import spinalML.dtypes.I8

// Top-Level Accelerator exposing standard AXI4-Stream interfaces for integration
// with physical ARM processors via DMA.
case class MLAccelerator(axiDataWidth: Int = 32) extends Component {
  // Let's configure our internal Tensor architecture
  // For this demo, we'll use a simple element-wise Mul operation on I8 tensors
  val dataType = I8()
  val shape = Seq(4) // e.g. a small vector
  val lanes = 2      // 2 elements processed per clock cycle
  
  val io = new Bundle {
    // Standard AXI4-Stream inputs (e.g. connected to DMA Read channels)
    val axisInA = slave(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))
    val axisInB = slave(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))
    
    // Standard AXI4-Stream output (e.g. connected to DMA Write channel)
    val axisOut = master(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))
  }
  
  // 1. Convert AXI4-Stream inputs to internal Tensor streams
  val convInA = Axi4StreamToTensor(dataType, shape, lanes, axiDataWidth)
  convInA.io.axis << io.axisInA
  
  val convInB = Axi4StreamToTensor(dataType, shape, lanes, axiDataWidth)
  convInB.io.axis << io.axisInB
  
  // 2. Instantiate the math operation (Core AI Pipeline)
  // We use our pipelined MulOp
  val mulOp = MulOp(dataType, shape, lanes)
  mulOp.io.a <> convInA.io.tensor
  mulOp.io.b <> convInB.io.tensor
  
  // 3. Convert internal Tensor stream back to AXI4-Stream output
  val convOut = TensorToAxi4Stream(dataType, shape, lanes, axiDataWidth)
  convOut.io.tensor <> mulOp.io.c
  
  io.axisOut << convOut.io.axis
}
