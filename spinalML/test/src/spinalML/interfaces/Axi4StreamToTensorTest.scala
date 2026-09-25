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
    SimConfig.withWave.compile(Axi4StreamToTensor(I8(), Seq(8), lanes = 2, axiDataWidth = 32)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.axis.valid #= false
      dut.io.axis.last #= false
      dut.io.tensor.stream.ready #= true
      dut.clockDomain.waitSampling()

      // One AXI beat, 32-bit bus, 2 lanes of 8 bits: [Lane 1 | Lane 0].
      // Values 5 and -10 (-10 = 0xF6, 5 = 0x05) pack as 0xF605.
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
