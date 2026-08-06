package spinalML.examples

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component to instantiate the pipeline with I8 data type
case class SimplePipelineTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(1, 2), lanes = 2))
    val b = slave(Tensor(I8(), Seq(1, 2), lanes = 2))
    val w = slave(Tensor(I8(), Seq(2, 1), lanes = 2))
    val y = master(Tensor(I8(), Seq(1, 1), lanes = 1))
  }
  
  val pipeline = SimplePipeline(I8(), lanes = 2)
  pipeline.io.a <> io.a
  pipeline.io.b <> io.b
  pipeline.io.w <> io.w
  io.y <> pipeline.io.y
}

class SimplePipelineTest extends AnyFunSuite {
  test("Test complete pipeline Y = Matmul(A + B, W) on I8 tensors") {
    SimConfig.withWave.compile(SimplePipelineTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize streams
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load Weights W into the Matmul Double-Buffer
      // W = [3, 4]T
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 3
      dut.io.w.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      
      dut.io.w.stream.valid #= false
      
      // Step 2: Stream Activations A and B concurrently
      // A = [1, 2]
      // B = [10, 20]
      // Expected sum = A + B = [11, 22]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 2
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 10
      dut.io.b.stream.payload(1) #= 20
      
      // Both streams should be consumed together due to StreamJoin in add.scala
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean && dut.io.b.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      
      // Step 3: Wait for pipeline output Y
      // Y = sum * W = [11, 22] * [3, 4]T
      // Y = 11*3 + 22*4 = 33 + 88 = 121
      dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean)
      
      val result = dut.io.y.stream.payload(0).toInt
      assert(result == 121, s"Expected 121, got $result")
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
