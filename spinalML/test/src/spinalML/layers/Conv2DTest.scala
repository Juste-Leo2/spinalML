package spinalML.layers

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16, I32}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class Conv2DTestComp[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc]) extends Component {
  val H = 3
  val W_in = 3
  val K = 2
  val totalWindows = 4
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(H, W_in), lanes = 1)) // 3x3 Image
    val w = slave(Tensor(dataType, Seq(K * K, 1), lanes = K * K)) // 2x2 Kernel Flattened
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(totalWindows, 1), lanes = 1)) // Output Flattened
  }
  io.y <> Conv2D(io.x, io.w, io.b, accType)
}

class Conv2DTest extends AnyFunSuite {
  test("Test Conv2D Layer: Y = Conv2D(X, W) + b on I4") {
    SimConfig.withWave.compile(Conv2DTestComp(I4(), I16())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.x.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // 1. Send Weights W = [1, 0, 0, 1] (Identity-like matrix to trace easily)
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 1
      dut.io.w.stream.payload(1) #= 0
      dut.io.w.stream.payload(2) #= 0
      dut.io.w.stream.payload(3) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // 2. Send Bias b = 2
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // 3. Send Input X = [-4..4] -> [-4, -3, -2, -1, 0, 1, 2, 3, 4]
      val inputs = Seq(-4, -3, -2, -1, 0, 1, 2, 3, 4)
      // windows are:
      // w1: [-4, -3, -1, 0] -> -4*1 + 0*1 + 2 = -2
      // w2: [-3, -2, 0, 1] -> -3*1 + 1*1 + 2 = 0
      // w3: [-1, 0, 2, 3] -> -1*1 + 3*1 + 2 = 4
      // w4: [0, 1, 3, 4] -> 0*1 + 4*1 + 2 = 6
      val expected = Seq(-2, 0, 4, 6)
      
      fork {
        var i = 0
        dut.io.x.stream.valid #= true
        while (i < 9) {
          dut.io.x.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSampling()
          if (dut.io.x.stream.ready.toBoolean) i += 1
        }
        dut.io.x.stream.valid #= false
      }
      
      // 4. Verify Output Y
      var i = 0
      while (i < 4) {
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

  test("Test Conv2D compilation on I16") {
    SpinalConfig().generateVerilog(Conv2DTestComp(I16(), I32()))
  }

  test("Test Conv2D compilation on FP4") {
    SpinalConfig().generateVerilog(Conv2DTestComp(FP4_E2M1(), FP4_E2M1()))
  }

  test("Test Conv2D compilation on BF16") {
    SpinalConfig().generateVerilog(Conv2DTestComp(BF16(), BF16()))
  }
}
