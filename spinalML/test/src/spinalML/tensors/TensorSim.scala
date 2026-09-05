// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.tensors

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.dtypes._

class TensorTestModule extends Component {
  val io = new Bundle {
    val t2d = master(Tensor.Tensor2D(I8(), ne0 = 2, ne1 = 3, lanes = 2))
    val t1d = master(Tensor.Tensor1D(U4(), ne0 = 4, lanes = 2))
  }

  // Drive default values to valid streams to satisfy the compiler
  io.t2d.stream.valid := True
  io.t2d.stream.payload(0) := 10
  io.t2d.stream.payload(1) := -5
  
  io.t1d.stream.valid := True
  io.t1d.stream.payload(0) := 0
  io.t1d.stream.payload(1) := 5
}

class TensorSim extends AnyFunSuite {
  test("Tensor streaming instantiation") {
    SimConfig.withWave.compile(new TensorTestModule).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.t2d.stream.ready #= true
      dut.io.t1d.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Test first chunk values
      assert(dut.io.t2d.stream.payload(0).toInt == 10)
      assert(dut.io.t2d.stream.payload(1).toInt == -5)
      
      assert(dut.io.t1d.stream.payload(0).toInt == 0)
      assert(dut.io.t1d.stream.payload(1).toInt == 5)
    }
  }
}
