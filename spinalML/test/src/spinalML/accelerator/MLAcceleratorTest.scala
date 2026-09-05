// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.accelerator

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import org.scalatest.funsuite.AnyFunSuite

class MLAcceleratorTest extends AnyFunSuite {
  test("Test MLAccelerator top-level with AXI4-Stream inputs and outputs") {
    SimConfig.withWave.compile(MLAccelerator(axiDataWidth = 32)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Init
      dut.io.axisInA.valid #= false
      dut.io.axisInA.last #= false
      dut.io.axisInB.valid #= false
      dut.io.axisInB.last #= false
      dut.io.axisOut.ready #= true
      
      dut.clockDomain.waitSampling()
      
      var seenData = scala.collection.mutable.ArrayBuffer[BigInt]()
      
      // Monitor output AXI Stream
      StreamMonitor(dut.io.axisOut, dut.clockDomain) { payload =>
        seenData += payload.data.toBigInt
      }
      
      // Send data to axisInA and axisInB simultaneously (MulOp needs both)
      // We will send 1 chunk (2 elements)
      // A: [3, 4] -> 0x0403
      // B: [2, 5] -> 0x0502
      // Expected C: [6, 20] -> 0x1406
      
      dut.io.axisInA.valid #= true
      dut.io.axisInA.data #= 0x0403
      
      dut.io.axisInB.valid #= true
      dut.io.axisInB.data #= 0x0502
      
      // Wait for handshake on both
      dut.clockDomain.waitSamplingWhere(dut.io.axisInA.ready.toBoolean && dut.io.axisInB.ready.toBoolean)
      
      dut.io.axisInA.valid #= false
      dut.io.axisInB.valid #= false
      
      // Wait for output propagation (pipeline depth is 1 or 2 cycles)
      dut.clockDomain.waitSampling(5)
      
      assert(seenData.nonEmpty, "No data was output on AXI bus!")
      assert(seenData(0) == 0x1406, s"Expected 0x1406 but got 0x${seenData(0).toString(16)}")
    }
  }
}
