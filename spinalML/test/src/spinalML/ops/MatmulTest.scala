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
    }
  }
}
