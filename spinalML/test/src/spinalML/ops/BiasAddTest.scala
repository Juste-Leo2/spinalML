// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component (lanes = 1, scalar bias)
case class BiasAddTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 1), lanes = 1))
    val b = slave(Tensor(dataType, Seq(1, 1), lanes = 1))
    val c = master(Tensor(dataType, Seq(4, 1), lanes = 1))
  }
  io.c <> bias_add(io.a, io.b)
}

// Wrapper component (lanes = 2, 2-column bias broadcast)
case class BiasAddTestComp2[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(1, 2), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 2))
  }
  io.c <> bias_add(io.a, io.b)
}

class BiasAddTest extends AnyFunSuite {
  test("Test broadcast addition of bias on I4 stream") {
    SimConfig.withWave.compile(BiasAddTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Send the bias
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // Step 2: Send the 4 elements of A
      val inputs = Seq(1, -3, 3, -4)
      var expectedOutputs = inputs.map(_ + 2)
      
      for(i <- 0 until 4) {
        dut.io.a.stream.valid #= true
        dut.io.a.stream.payload(0) #= inputs(i)
        
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean && dut.io.c.stream.valid.toBoolean)
        
        val result = dut.io.c.stream.payload(0).toInt
        assert(result == expectedOutputs(i), s"Expected ${expectedOutputs(i)}, got $result")
      }
      
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
    test(s"Test BiasAdd compilation on $name") {
      SpinalConfig().generateVerilog(BiasAddTestComp(dt()))
      SpinalConfig().generateVerilog(BiasAddTestComp2(dt()))
    }
  }
}
