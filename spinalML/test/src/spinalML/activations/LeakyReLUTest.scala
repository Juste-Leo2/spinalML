// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the LeakyReLU operation
case class LeakyReLUTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4), lanes = 2))
    val y = master(Tensor(dataType, Seq(4), lanes = 2))
  }
  
  // shift = 1 means multiplying by 0.5
  io.y <> spinalML.activations.leaky_relu(io.x, shift = 1)
}

class LeakyReLUTest extends AnyFunSuite {
  test("Test streaming LeakyReLU operation on I4 tensors") {
    SimConfig.withWave.compile(LeakyReLUTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.x.stream.valid #= true
      dut.io.x.stream.payload(0) #= 6
      dut.io.x.stream.payload(1) #= -4
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 1
      assert(dut.io.y.stream.payload(0).toInt == 6)
      // -4 >> 1 = -2
      assert(dut.io.y.stream.payload(1).toInt == -2)
      
      // Send second chunk
      dut.io.x.stream.payload(0) #= -6
      dut.io.x.stream.payload(1) #= 3
      
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean)
      
      // Check results for chunk 2
      // -6 >> 1 = -3
      assert(dut.io.y.stream.payload(0).toInt == -3)
      assert(dut.io.y.stream.payload(1).toInt == 3)
      
      dut.io.x.stream.valid #= false
      
      dut.clockDomain.waitSampling(5)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test LeakyReLU compilation on $name") {
      SpinalConfig().generateVerilog(LeakyReLUTestComp(dt()))
    }
  }
}
