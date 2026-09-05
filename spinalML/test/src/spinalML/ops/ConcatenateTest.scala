// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Concatenate Axis 0
case class ConcatenateTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4), lanes = 2))
    val c = master(Tensor(dataType, Seq(6), lanes = 2))
  }
  
  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 0)
}

class ConcatenateTest extends AnyFunSuite {
  test("Test streaming Concatenate operation on Axis 0") {
    SimConfig.withWave.compile(ConcatenateTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed A
      fork {
        for(i <- 0 until 2) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= i + 1
          dut.io.a.stream.payload(1) #= i + 2
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        }
        dut.io.a.stream.valid #= false
      }
      
      // Feed B (concurrently, should be blocked until A is done)
      fork {
        for(i <- 0 until 4) {
          dut.io.b.stream.valid #= true
          dut.io.b.stream.payload(0) #= i + 3
          dut.io.b.stream.payload(1) #= i + 4
          dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
        }
        dut.io.b.stream.valid #= false
      }
      
      // Check results
      for(i <- 0 until 2) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
        assert(dut.io.c.stream.payload(0).toInt == i + 1)
      }
      for(i <- 0 until 4) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
        assert(dut.io.c.stream.payload(0).toInt == i + 3)
      }
      
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
    test(s"Test Concatenate compilation on $name") {
      SpinalConfig().generateVerilog(ConcatenateTestComp(dt()))
    }
  }
}
