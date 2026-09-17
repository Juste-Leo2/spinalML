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

// OPS-04: 2D geometry where one axis-0 row spans several beats
// ([2,3] ++ [3,3] at lanes=1 -> [5,3], 6 + 9 = 15 beats out).
case class ConcatenateTest2D[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 3), lanes = 1))
    val b = slave(Tensor(dataType, Seq(3, 3), lanes = 1))
    val c = master(Tensor(dataType, Seq(5, 3), lanes = 1))
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

  // OPS-04 red test: the FSM must count streamed beats (rows x
  // beats-per-row), not shape.head cells.
  test("OPS-04 Concatenate axis0 2D lanes=1 streams all 15 beats in order") {
    SimConfig.withWave.compile(ConcatenateTest2D(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // A rows [1,2,3],[4,5,6] then B rows [7..15], one element per beat.
      // Bounded waits: a miscount becomes a clean failure, never a hang.
      // NOTE: the op has no output buffer (c.valid is combinational on the
      // inputs), so outputs are sampled as each input beat is accepted.
      var got = Vector.empty[Int]
      def drive(v: Int, tag: String, bi: Int): Unit = {
        if (tag == "A") dut.io.a.stream.payload(0) #= v
        else dut.io.b.stream.payload(0) #= v
        val accepted = if (tag == "A")
          dut.clockDomain.waitSamplingWhere(100)(dut.io.a.stream.ready.toBoolean)
        else
          dut.clockDomain.waitSamplingWhere(100)(dut.io.b.stream.ready.toBoolean)
        assert(!accepted, s"$tag beat $bi never accepted")
        if (dut.io.c.stream.valid.toBoolean) got = got :+ dut.io.c.stream.payload(0).toInt
      }
      val aData = Seq(1, 2, 3, 4, 5, 6)
      dut.io.a.stream.valid #= true
      for ((v, bi) <- aData.zipWithIndex) drive(v, "A", bi)
      dut.io.a.stream.valid #= false

      val bData = Seq(7, 8, 9, 10, 11, 12, 13, 14, 15)
      dut.io.b.stream.valid #= true
      for ((v, bi) <- bData.zipWithIndex) drive(v, "B", bi)
      dut.io.b.stream.valid #= false

      val expected = aData ++ bData
      assert(got == expected, s"expected $expected, got $got")
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
