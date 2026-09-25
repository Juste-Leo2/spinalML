// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

case class CumsumTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    // 3 rows (L=3), 2 cols (C=2). Data streams 2 elements per cycle (lanes=2)
    val in = slave(Tensor(dataType, Seq(3, 2), lanes = 2))
    val out = master(Tensor(dataType, Seq(3, 2), lanes = 2))
  }
  
  io.out <> spinalML.ops.cumsum(io.in)
}

// C = 5 with lanes = 4: each row occupies ceil(C/lanes) = 2 padded beats
// (project-wide per-row padded stream contract, same as MatmulDynPad).
case class CumsumUnalignedTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val in = slave(Tensor(dataType, Seq(2, 5), lanes = 4))
    val out = master(Tensor(dataType, Seq(2, 5), lanes = 4))
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
      
      // L=0, (1, 2)
      dut.io.in.stream.valid #= true
      dut.io.in.stream.payload(0) #= 1
      dut.io.in.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
      
      // L=1, (3, 4)
      dut.io.in.stream.payload(0) #= 3
      dut.io.in.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
      
      // L=2, (5, 6)
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

  test("CumSum with C=5, lanes=4 on per-row padded beats") {
    SimConfig.withWave.compile(CumsumUnalignedTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.in.stream.valid #= false
      dut.io.out.stream.ready #= true
      dut.clockDomain.waitSampling()

      val inBeats = Seq(
        Seq(1, 2, 3, 4),
        Seq(5, 0, 0, 0),
        Seq(10, 20, 30, 40),
        Seq(50, 0, 0, 0)
      )
      val expected = Seq(
        Seq(1, 2, 3, 4),
        Seq(5, 0, 0, 0),
        Seq(11, 22, 33, 44),
        Seq(55, 0, 0, 0)
      )

      fork {
        for (b <- inBeats) {
          dut.io.in.stream.valid #= true
          for (i <- 0 until 4) dut.io.in.stream.payload(i) #= b(i)
          dut.clockDomain.waitSamplingWhere(dut.io.in.stream.ready.toBoolean)
        }
        dut.io.in.stream.valid #= false
      }

      for (b <- expected) {
        dut.clockDomain.waitSamplingWhere(dut.io.out.stream.valid.toBoolean)
        for (i <- 0 until 4) {
          assert(dut.io.out.stream.payload(i).toInt == b(i),
            s"lane $i: got ${dut.io.out.stream.payload(i).toInt}, expected ${b(i)}")
        }
      }
      dut.clockDomain.waitSampling(2)
    }
  }
}
