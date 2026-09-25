// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.accelerator

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axis._
import spinalML.interfaces._
import spinalML.ops._
import spinalML.dtypes.I8

// Demo top-level: AXI4-Stream in/out around an element-wise MulOp on I8 tensors.
case class MLAccelerator(axiDataWidth: Int = 32) extends Component {
  val dataType = I8()
  val shape = Seq(4)
  val lanes = 2
  
  val io = new Bundle {
    val axisInA = slave(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))
    val axisInB = slave(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))

    val axisOut = master(Axi4Stream(Axi4StreamConfig(dataWidth = axiDataWidth / 8, useLast = true)))
  }

  val convInA = Axi4StreamToTensor(dataType, shape, lanes, axiDataWidth)
  convInA.io.axis << io.axisInA
  
  val convInB = Axi4StreamToTensor(dataType, shape, lanes, axiDataWidth)
  convInB.io.axis << io.axisInB
  
  val mulOp = MulOp(dataType, shape, lanes)
  mulOp.io.a <> convInA.io.tensor
  mulOp.io.b <> convInB.io.tensor

  val convOut = TensorToAxi4Stream(dataType, shape, lanes, axiDataWidth)
  convOut.io.tensor <> mulOp.io.c
  
  io.axisOut << convOut.io.axis
}
