// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16, I32}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class Conv1DTestComp[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(3, 1), lanes = 1)) // Sequence of 3
    val w = slave(Tensor(dataType, Seq(2, 1), lanes = 2)) // K=2, C=1, M=1 (Weight)
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    
    val y = master(Tensor(accType, Seq(2, 1), lanes = 1))
  }

  io.y <> Conv1D(io.x, io.w, io.b, accType, parallelN = false)
}

case class Conv1DTestCompMulti[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(3, 2), lanes = 1)) // Sequence of 3, C=2, lanes=1
    val w = slave(Tensor(dataType, Seq(4, 2), lanes = 4)) // K=2, inC=2, outC=2. Total w shape = [4, 2]. lanes=4 (tileSize = 4)
    val b = slave(Tensor(accType, Seq(1, 2), lanes = 1)) // Bias for outC=2, lanes=1
    
    val y = master(Tensor(accType, Seq(2, 2), lanes = 1)) // Output seq 2, outC=2, lanes=1
  }

  io.y <> Conv1D(io.x, io.w, io.b, accType, parallelN = false)
}

class Conv1DTest extends AnyFunSuite {
  test("Test Conv1D Layer: Y = Conv1D(X, W) + b on I4") {
    SimConfig.withWave.compile(Conv1DTestComp(I4(), I16())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.x.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // 1. Send Weights W = [2, -1] (lanes = 2, so sent in 1 cycle)
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 2
      dut.io.w.stream.payload(1) #= -1
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // 2. Send Bias b = 3
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // 3. Send Input X = [1, 2, 3]
      // expected:
      // win1 = [1, 2] * [2, -1] + 3 = 2 - 2 + 3 = 3
      // win2 = [2, 3] * [2, -1] + 3 = 4 - 3 + 3 = 4
      val inputs = Seq(1, 2, 3)
      val expected = Seq(3, 4)
      
      fork {
        var i = 0
        dut.io.x.stream.valid #= true
        while (i < 3) {
          dut.io.x.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSampling()
          if (dut.io.x.stream.ready.toBoolean) i += 1
        }
        dut.io.x.stream.valid #= false
      }
      
      // 4. Verify Output Y
      var i = 0
      while (i < 2) {
        dut.clockDomain.waitSampling()
        if (dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean) {
          val result = dut.io.y.stream.payload(0).toInt
          assert(result == expected(i), s"Output $i: expected ${expected(i)}, got $result")
          i += 1
        }
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
    val accDt = if (name == "I8" || name == "I16") () => I32() else dt
    test(s"Test Conv1D compilation on $name") {
      SpinalConfig().generateVerilog(Conv1DTestComp(dt(), accDt()))
    }
    test(s"Test Conv1DMulti compilation on $name") {
      SpinalConfig().generateVerilog(Conv1DTestCompMulti(dt(), accDt()))
    }
  }
}
