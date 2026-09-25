// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

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

// OPS-05: axis-1 joins whole rows side-by-side per beat.
case class ConcatenateAxis1_2D[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 4), lanes = 4))
  }

  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 1)
}

// OPS-05: sub-row lanes would interleave half-rows instead of joining rows.
case class ConcatenateAxis1SubRow[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2, 4), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 8), lanes = 4))
  }

  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 1)
}

// OPS-05: unequal lanes deadlock the lockstep StreamJoin (beat counts differ).
case class ConcatenateAxis1LanesMismatch[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), lanes = 4))
    val b = slave(Tensor(dataType, Seq(2, 4), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 8), lanes = 6))
  }

  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 1)
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

  // OPS-05: the row-wise geometry (lanes == row width) joins rows correctly.
  test("OPS-05 Concatenate axis1 2D lanes=rowWidth joins rows side-by-side") {
    SimConfig.withWave.compile(ConcatenateAxis1_2D(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // NOTE: the join is lockstep (both beats each cycle); valids drop for
      // one cycle between rows so each output beat is sampled exactly once.
      // Bounded waits throughout: a stall becomes a clean failure.
      val rowsA = Seq(Seq(1, 2), Seq(3, 4))
      val rowsB = Seq(Seq(5, 6), Seq(7, 8))
      var got = Vector.empty[Seq[Int]]
      for ((ra, rb) <- rowsA.zip(rowsB)) {
        dut.io.a.stream.payload(0) #= ra(0)
        dut.io.a.stream.payload(1) #= ra(1)
        dut.io.b.stream.payload(0) #= rb(0)
        dut.io.b.stream.payload(1) #= rb(1)
        dut.io.a.stream.valid #= true
        dut.io.b.stream.valid #= true
        assert(!dut.clockDomain.waitSamplingWhere(100)(dut.io.c.stream.valid.toBoolean),
          s"row $ra ++ $rb never joined")
        got = got :+ (0 until 4).map(i => dut.io.c.stream.payload(i).toInt)
        dut.io.a.stream.valid #= false
        dut.io.b.stream.valid #= false
        dut.clockDomain.waitSampling()
      }
      val expected = Seq(Seq(1, 2, 5, 6), Seq(3, 4, 7, 8))
      assert(got == expected, s"expected $expected, got $got")
      dut.clockDomain.waitSampling(5)
    }
  }

  test("OPS-05 Concatenate axis1 rejects lanes != row width and lanesA != lanesB") {
    // [2,4] at lanes=2: beats hold half-rows; the per-beat join would emit
    // interleaved half-joins instead of concatenated rows. Repack first.
    assertThrows[IllegalArgumentException] {
      spinal.core.SpinalVerilog(ConcatenateAxis1SubRow(I8()))
    }
    // lanes 4 vs 2 on the same shape: different beat counts deadlock the
    // lockstep StreamJoin (one side runs dry).
    assertThrows[IllegalArgumentException] {
      spinal.core.SpinalVerilog(ConcatenateAxis1LanesMismatch(I8()))
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
