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
  val L_in = 3
  val K = 2
  val L_out = 2
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L_in, 1), lanes = 1)) // Input Sequence
    val w = slave(Tensor(dataType, Seq(K, 1), lanes = K)) // Kernel Weights MUST match seq2col lanes
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(L_out, 1), lanes = 1)) // Output Sequence
  }
  io.y <> Conv1D(io.x, io.w, io.b, accType)
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
    test(s"Test Conv1D compilation on $name") {
      SpinalConfig().generateVerilog(Conv1DTestComp(dt(), dt()))
    }
  }
}
