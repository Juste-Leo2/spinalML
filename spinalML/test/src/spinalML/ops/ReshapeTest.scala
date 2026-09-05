// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Reshape (and indirectly Flatten)
case class ReshapeTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), lanes = 2)) // 8 elements total
    val reshaped = master(Tensor(dataType, Seq(4, 2), lanes = 2))
  }
  
  // We can chain them to test both metadata ops
  val flat = spinalML.ops.flatten(io.a)
  val resh = spinalML.ops.reshape(flat, Seq(4, 2))
  
  io.reshaped.stream << resh.stream
}

class ReshapeTest extends AnyFunSuite {
  test("Test Reshape and Flatten metadata operations on I8 tensors") {
    SimConfig.withWave.compile(ReshapeTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize
      dut.io.a.stream.valid #= false
      dut.io.reshaped.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed data
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 2
      
      dut.clockDomain.waitSamplingWhere(dut.io.reshaped.stream.valid.toBoolean && dut.io.reshaped.stream.ready.toBoolean)
      
      assert(dut.io.reshaped.stream.payload(0).toInt == 1)
      assert(dut.io.reshaped.stream.payload(1).toInt == 2)
      
      dut.io.a.stream.valid #= false
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
    test(s"Test Reshape compilation on $name") {
      SpinalConfig().generateVerilog(ReshapeTestComp(dt()))
    }
  }
}
