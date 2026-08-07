package spinalML.layers

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I32}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class Conv1DTestComp() extends Component {
  val L_in = 3
  val K = 2
  val L_out = 2
  val dataType = I8()
  val accType = I32()
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L_in, 1), lanes = 1)) // Input Sequence
    val w = slave(Tensor(dataType, Seq(K, 1), lanes = K)) // Kernel Weights MUST match seq2col lanes
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // Bias
    val y = master(Tensor(accType, Seq(L_out, 1), lanes = 1)) // Output Sequence
  }
  io.y <> Conv1D(io.x, io.w, io.b, accType)
}

class Conv1DTest extends AnyFunSuite {
  test("Test Conv1D Layer: Y = Conv1D(X, W) + b") {
    SimConfig.withWave.compile(Conv1DTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.x.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // 1. Send Weights W = [2, 3] (lanes = 2, so sent in 1 cycle)
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 2
      dut.io.w.stream.payload(1) #= 3
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // 2. Send Bias b = 5
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 5
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // 3. Send Input X = [10, 20, 30]
      val inputs = Seq(10, 20, 30)
      val expected = Seq(85, 135)
      
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
}
