package spinalML.layers

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class LinearTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(1, 2), lanes = 2)) // 1 row, 2 cols (M=1, K=2)
    val w = slave(Tensor(I8(), Seq(2, 1), lanes = 2)) // 2 weights (K=2, 1)
    val b = slave(Tensor(I8(), Seq(1, 1), lanes = 1)) // 1 bias
    val y = master(Tensor(I8(), Seq(1, 1), lanes = 1))
  }
  io.y <> Linear(io.a, io.w, io.b, tileSize = 2)
}

class LinearTest extends AnyFunSuite {
  test("Test Linear Layer: Y = A * W + b on small I8 tensors") {
    SimConfig.withWave.compile(LinearTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load Weights W into the Matmul Double-Buffer
      // W = [3, 4]T
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 3
      dut.io.w.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // Step 2: Send the bias b
      // b = 5
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 5
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // Step 3: Stream Activations A (1 row)
      // Row 0: [1, 2] -> Y0 = (1*3 + 2*4) + 5 = 11 + 5 = 16
      
      val a_data = Seq(Seq(1, 2))
      val expected_y = Seq(16)
      
      // Start streaming A
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= a_data(0)(0)
      dut.io.a.stream.payload(1) #= a_data(0)(1)
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 4: Verify output Y (1 row)
      for (i <- 0 until 1) {
        dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean)
        val result = dut.io.y.stream.payload(0).toInt
        assert(result == expected_y(i), s"Expected ${expected_y(i)} for row $i, got $result")
        // Since y streams at 1 element per cycle, wait 1 cycle to consume
        dut.clockDomain.waitSampling(1)
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
