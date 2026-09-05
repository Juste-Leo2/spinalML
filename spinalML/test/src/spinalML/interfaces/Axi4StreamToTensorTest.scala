// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.interfaces

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib.bus.amba4.axis._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

class Axi4StreamToTensorTest extends AnyFunSuite {
  test("Test conversion from standard AXI4-Stream to Tensor stream") {
    // We create a Component dynamically to test the converter
    SimConfig.withWave.compile(Axi4StreamToTensor(I8(), Seq(8), lanes = 2, axiDataWidth = 32)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Init
      dut.io.axis.valid #= false
      dut.io.axis.last #= false
      dut.io.tensor.stream.ready #= true
      dut.clockDomain.waitSampling()
      
      // Let's send one beat over AXI
      // AXI Data Width is 32 bits. Tensor lanes is 2 (2 * 8 bits = 16 bits).
      // AXI bits: [Lane 1 (8 bits) | Lane 0 (8 bits)]
      // E.g., we want to send values 5 and -10.
      // -10 in 8-bit hex: F6. 5 in 8-bit hex: 05.
      // Packed: 0xF605
      
      // We want to test that the tensor stream outputs the correct payload
      // We can use a StreamMonitor or just check it directly during the handshake
      var seenPayload0 = 0
      var seenPayload1 = 0
      
      StreamReadyRandomizer(dut.io.tensor.stream, dut.clockDomain)
      StreamMonitor(dut.io.tensor.stream, dut.clockDomain) { payload =>
        seenPayload0 = payload(0).toInt
        seenPayload1 = payload(1).toInt
      }
      
      dut.io.axis.valid #= true
      dut.io.axis.data #= 0xF605
      dut.clockDomain.waitSamplingWhere(dut.io.axis.ready.toBoolean)
      
      dut.io.axis.valid #= false
      dut.clockDomain.waitSampling(5)
      
      assert(seenPayload0 == 5)
      assert(seenPayload1 == -10)
    }
  }
}
