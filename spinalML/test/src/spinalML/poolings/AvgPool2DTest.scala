// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.poolings

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import spinalML.{RoundingConfig, RoundingMode}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the AvgPool2D operation: 4x4 -> 2x2 (poolSize = 2, stride = 2)
case class AvgPool2DTestComp[T <: Data](dataType: HardType[T], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 4), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 1))
  }

  io.c <> spinalML.poolings.avgpool2d(io.a, poolSize = 2, stride = 2, rounding = rounding)
}

// Overlapping windows: 3x3 -> 2x2 (poolSize = 2, stride = 1)
case class AvgPool2DTestCompStride1[T <: Data](dataType: HardType[T], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(3, 3), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 1))
  }

  io.c <> spinalML.poolings.avgpool2d(io.a, poolSize = 2, stride = 1, rounding = rounding)
}

// Multi-channel: 4x4x2 -> 2x2x2 (poolSize = 2, stride = 2)
case class AvgPool2DTestCompMulti[T <: Data](dataType: HardType[T], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 4, 2), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 2, 2), lanes = 2))
  }

  io.c <> spinalML.poolings.avgpool2d(io.a, poolSize = 2, stride = 2, rounding = rounding)
}

// ACT-03 repro: 5x5, pool=2, stride=2 -> 2x2 windows leave a 6-beat tail per
// frame that must be drained before the next frame starts.
case class AvgPool2DResidueTestComp[T <: Data](dataType: HardType[T], rounding: RoundingMode = RoundingConfig.current) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(5, 5), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 1))
  }

  io.c <> spinalML.poolings.avgpool2d(io.a, poolSize = 2, stride = 2, rounding = rounding)
}

class AvgPool2DTest extends AnyFunSuite {
  // 4x4 image, windows 2x2 stride 2 with two RNE ties:
  // [  1   2   3   4]   avg(1,2,5,6)   = 14/4 = 3.5 -> RNE 4 (floor 3 odd), trunc 3
  // [  5   6   7   8]   avg(3,4,7,8)   = 22/4 = 5.5 -> RNE 6 (floor 5 odd), trunc 5
  // [ -1  -2  -3  -4]   avg(-1,-2,9,10)= 16/4 = 4 exact
  // [  9  10  11  12]   avg(-3,-4,11,12)= 16/4 = 4 exact
  private def runAvgPool2DShift(rounding: RoundingMode, expected: Seq[Int]): Unit = {
    SimConfig.compile(AvgPool2DTestComp(I8(), rounding)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true

      dut.clockDomain.waitSampling()

      val inputs = Seq(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 9, 10, 11, 12)
      var i = 0

      fork {
        while (i < 16) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          i += 1
        }
        dut.io.a.stream.valid #= false
      }

      var o = 0
      while (o < 4) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
        val result = dut.io.c.stream.payload(0).toInt
        assert(result == expected(o), s"Output $o: expected ${expected(o)}, got $result (rounding=$rounding)")
        o += 1
      }

      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test streaming AvgPool2D operation on I8 tensors (RNE)") {
    runAvgPool2DShift(RoundingMode.Rne, Seq(4, 6, 4, 4))
  }

  test("Test streaming AvgPool2D operation on I8 tensors (Truncate legacy)") {
    runAvgPool2DShift(RoundingMode.Truncate, Seq(3, 5, 4, 4))
  }

  test("Test streaming AvgPool2D stride 1 on I8 tensors") {
    SimConfig.withWave.compile(AvgPool2DTestCompStride1(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true

      dut.clockDomain.waitSampling()

      // 3x3 image, windows 2x2 stride 1 (overlapping)
      // avg(1,2,4,5)=12>>2=3, avg(2,3,5,6)=16>>2=4, avg(4,5,7,8)=24>>2=6, avg(5,6,8,9)=28>>2=7
      val inputs = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9)
      val expected = Seq(3, 4, 6, 7)
      var i = 0

      fork {
        while (i < 9) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          i += 1
        }
        dut.io.a.stream.valid #= false
      }

      var o = 0
      while (o < 4) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
        val result = dut.io.c.stream.payload(0).toInt
        assert(result == expected(o), s"Output $o: expected ${expected(o)}, got $result")
        o += 1
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
    test(s"Test AvgPool2D compilation on $name") {
      SpinalConfig().generateVerilog(AvgPool2DTestComp(dt()))
    }
    test(s"Test AvgPool2DStride1 compilation on $name") {
      SpinalConfig().generateVerilog(AvgPool2DTestCompStride1(dt()))
    }
    test(s"Test AvgPool2DMulti compilation on $name") {
      SpinalConfig().generateVerilog(AvgPool2DTestCompMulti(dt()))
    }
  }

  test("Test AvgPool2D residue compilation on I8") {
    SpinalConfig().generateVerilog(AvgPool2DResidueTestComp(I8()))
  }
}
