// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing matmul: Matrix A [1, 2] x Vector B [2, 1]
case class MatmulTest_Vector[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2, 1), lanes = 2))
    val c = master(Tensor(dataType, Seq(1, 1), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = false)
}

// Component for testing GEMM Parallel: A[2, 4] x B[4, 2]
case class MatmulTest_GEMM_Parallel[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = true)
}

// Component for testing GEMM Sequential: A[2, 4] x B[4, 2]
case class MatmulTest_GEMM_Sequential[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), lanes = 2))
    val b = slave(Tensor(dataType, Seq(4, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = false)
}

// Component for testing Dynamic Padding: A[1, 3] x B[3, 1] with lanes=2
case class MatmulTest_DynamicPadding[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 3), lanes = 2))
    val b = slave(Tensor(dataType, Seq(3, 1), lanes = 2))
    val c = master(Tensor(dataType, Seq(1, 1), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = false)
}

// Component for the OPS-07 contract tests: A[1, 3] x B[3, 2] with lanes=2
case class MatmulTest_DynamicPaddingWide[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 3), lanes = 2))
    val b = slave(Tensor(dataType, Seq(3, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(1, 2), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = false)
}

// Component for testing Batched Matmul: A[2, 1, 2] x B[2, 2, 1]
case class MatmulTest_Batched[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 1, 2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2, 2, 1), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 1, 1), lanes = 1))
  }
  io.c <> spinalML.ops.matmul(io.a, io.b, parallelN = false)
}

class MatmulTest extends AnyFunSuite {
  test("Test streaming matmul Vector operation on I8 tensors") {
    SimConfig.withWave.compile(MatmulTest_Vector(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load matrix B into internal SRAM
      // B = [3, -2]T
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= -2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.valid #= false
      
      // Step 2: Stream Matrix A to compute
      // Row 0: [2, 1]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 2
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 3: Wait for output C
      // 2*3 + 1*(-2) = 6 - 2 = 4
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 4)
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test streaming matmul GEMM Parallel on I8") {
    SimConfig.withWave.compile(MatmulTest_GEMM_Parallel(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()
      
      // B is 4x2. K=4, N=2. chunksK = 2.
      dut.io.b.stream.valid #= true
      // Col 0, Chunk 0
      dut.io.b.stream.payload(0) #= 1
      dut.io.b.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      // Col 0, Chunk 1
      dut.io.b.stream.payload(0) #= 1
      dut.io.b.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      // Col 1, Chunk 0
      dut.io.b.stream.payload(0) #= 2
      dut.io.b.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      // Col 1, Chunk 1
      dut.io.b.stream.payload(0) #= 2
      dut.io.b.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.valid #= false
      
      // Stream A (2x4). 2 rows, 2 chunks per row.
      dut.io.a.stream.valid #= true
      // Row 0, Chunk 0
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 0
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      // Row 0, Chunk 1
      dut.io.a.stream.payload(0) #= 0
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      // Row 1, Chunk 0
      dut.io.a.stream.payload(0) #= 0
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      // Row 1, Chunk 1
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 0
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      var count = 0
      while(count < 4) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean) {
            count += 1
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
    test(s"Test Matmul compilation on $name") {
      SpinalConfig().generateVerilog(MatmulTest_Vector(dt()))
      SpinalConfig().generateVerilog(MatmulTest_GEMM_Parallel(dt()))
      SpinalConfig().generateVerilog(MatmulTest_GEMM_Sequential(dt()))
      SpinalConfig().generateVerilog(MatmulTest_DynamicPadding(dt()))
      SpinalConfig().generateVerilog(MatmulTest_Batched(dt()))
      SpinalConfig().generateVerilog(MatmulTest_DynamicPaddingWide(dt()))
    }
  }

  // OPS-07 contract guard: K=3, lanes=2 with per-line PADDED groups on both
  // inputs computes the exact result. A=[7,8,9], B col0=[1,2,3], col1=[4,5,6]:
  // C = [7+16+27, 28+40+54] = [50, 122].
  test("OPS-07 padded K=3 lanes=2 computes exact result (contract)") {
    SimConfig.withWave.compile(MatmulTest_DynamicPaddingWide(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // B, column-major, chunksK=2 beats per column, zero-padded tail lanes.
      // Bounded waits: a starved buffer becomes a clean failure, never an
      // infinite load.
      val bBeats = Seq(Seq(1, 2), Seq(3, 0), Seq(4, 5), Seq(6, 0))
      dut.io.b.stream.valid #= true
      for ((beat, bi) <- bBeats.zipWithIndex) {
        dut.io.b.stream.payload(0) #= beat(0)
        dut.io.b.stream.payload(1) #= beat(1)
        val bTimeout = dut.clockDomain.waitSamplingWhere(100)(dut.io.b.stream.ready.toBoolean)
        assert(!bTimeout, s"B beat $bi never accepted (B buffer starved)")
      }
      dut.io.b.stream.valid #= false

      // A row, chunksK=2 beats, zero-padded tail lane
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 7
      dut.io.a.stream.payload(1) #= 8
      assert(!dut.clockDomain.waitSamplingWhere(100)(dut.io.a.stream.ready.toBoolean), "A beat 0 never accepted")
      dut.io.a.stream.payload(0) #= 9
      dut.io.a.stream.payload(1) #= 0
      assert(!dut.clockDomain.waitSamplingWhere(100)(dut.io.a.stream.ready.toBoolean), "A beat 1 never accepted")
      dut.io.a.stream.valid #= false

      val expected = Seq(50, 122)
      var got = Vector.empty[Int]
      var cycles = 0
      while (got.length < 2 && cycles < 500) {
        dut.clockDomain.waitSampling()
        cycles += 1
        if (dut.io.c.stream.valid.toBoolean) got = got :+ dut.io.c.stream.payload(0).toInt
      }
      assert(got == expected, s"expected $expected, got $got after $cycles cycles")
      dut.clockDomain.waitSampling(5)
    }
  }

  // OPS-07 trap documentation: the same geometry with a DENSELY packed B
  // (K*N = 6 elements over 3 beats, no per-column padding) starves the B
  // buffer (it waits for N*chunksK = 4 beats) and no output ever comes out.
  // Dense delivery with K % lanes != 0 is out of contract by design.
  test("OPS-07 dense unpadded B with K=3 lanes=2 never produces output") {
    SimConfig.withWave.compile(MatmulTest_DynamicPaddingWide(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // Dense column-major elements [1,2,3,4,5,6] over 3 beats (one short).
      // NOTE: the A side is driven best-effort with bounded waits: the B
      // starvation means tileReady never comes, so A ready never asserts and
      // an unbounded waitSamplingWhere would hang the bench itself.
      val bBeats = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      dut.io.b.stream.valid #= true
      for (beat <- bBeats) {
        dut.io.b.stream.payload(0) #= beat(0)
        dut.io.b.stream.payload(1) #= beat(1)
        dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      }
      dut.io.b.stream.valid #= false

      // A row, padded (2 beats), best-effort delivery attempt
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 7
      dut.io.a.stream.payload(1) #= 8
      for (_ <- 0 until 50) {
        dut.clockDomain.waitSampling()
        if (dut.io.a.stream.ready.toBoolean) {
          dut.io.a.stream.payload(0) #= 9
          dut.io.a.stream.payload(1) #= 0
        }
      }
      dut.io.a.stream.valid #= false

      // Bounded watchdog: 300 cycles with the output ready, nothing may come out
      var sawValid = false
      for (_ <- 0 until 300) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean) sawValid = true
      }
      assert(!sawValid, "dense unpadded B produced output: OPS-07 contract violated")
      dut.clockDomain.waitSampling(5)
    }
  }
}
