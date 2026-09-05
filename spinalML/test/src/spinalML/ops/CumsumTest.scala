// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the CumSum operation
case class CumsumTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    // 3 rows (L=3), 2 cols (C=2). Data streams 2 elements per cycle (lanes=2)
    val in = slave(Tensor(dataType, Seq(3, 2), lanes = 2))
    val out = master(Tensor(dataType, Seq(3, 2), lanes = 2))
  }
  
  io.out <> spinalML.ops.cumsum(io.in)
}

class CumsumTest extends AnyFunSuite {
  test("Test streaming CumSum operation on I8 tensors") {
    SimConfig.withWave.compile(CumsumTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.in.stream.valid #= false
      dut.io.out.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: L=0, (1, 2)
      dut.io.in.stream.valid #= true
      dut.io.in.stream.payload(0) #= 1
      dut.io.in.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
      
      // Step 2: L=1, (3, 4)
      dut.io.in.stream.payload(0) #= 3
      dut.io.in.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
      
      // Step 3: L=2, (5, 6)
      dut.io.in.stream.payload(0) #= 5
      dut.io.in.stream.payload(1) #= 6
      dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
      
      dut.io.in.stream.valid #= false
      
      // Wait for the pipeline to finish (it should take exactly 3 cycles after valid)
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
    test(s"Test CumSum compilation on $name") {
      SpinalConfig().generateVerilog(CumsumTestComp(dt()))
    }
  }
}
