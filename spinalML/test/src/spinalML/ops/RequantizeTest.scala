package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I32, I8}
import org.scalatest.funsuite.AnyFunSuite

case class RequantizeTestComp(shift: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I32(), Seq(4), lanes = 4))
    val c = master(Tensor(I8(), Seq(4), lanes = 4))
  }
  
  io.c <> requantize(io.a, I8(), shift)
}

class RequantizeTest extends AnyFunSuite {
  test("Test streaming Requantize operation I32 -> I8 with shift and saturation") {
    SimConfig.withWave.compile(RequantizeTestComp(shift = 2)).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      // I32 inputs
      val inputData = Array(
        Array(100, -100, 1000, -1000), // Expected (shift=2): 25, -25, 250->127, -250->-128
        Array(0, 10, -10, 508)         // Expected (shift=2): 0, 2, -3, 127
      )
      
      val expectedOutputs = Array(
        Array(25, -25, 127, -128),
        Array(0, 2, -3, 127)
      )
      
      var outputIndex = 0
      val numExpectedOutputs = inputData.length
      
      // Monitor output
      fork {
        while (outputIndex < numExpectedOutputs) {
          dut.clockDomain.waitSampling()
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            val expected = expectedOutputs(outputIndex)
            for (i <- 0 until 4) {
              val outVal = dut.io.c.stream.payload(i).asInstanceOf[SInt].toInt
              assert(outVal == expected(i), s"Index $outputIndex Lane $i: Expected ${expected(i)}, got $outVal")
            }
            outputIndex += 1
          }
        }
      }
      
      // Drive inputs
      for (i <- 0 until inputData.length) {
        dut.io.a.stream.valid #= true
        for (lane <- 0 until 4) {
          dut.io.a.stream.payload(lane).asInstanceOf[SInt] #= inputData(i)(lane)
        }
        
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
        dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
      }
      
      // Wait for all outputs to be checked
      dut.clockDomain.waitSamplingWhere(outputIndex == numExpectedOutputs)
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Requantize compilation on I32_I8_shift2") {
    SpinalConfig().generateVerilog(RequantizeTestComp(shift = 2))
  }
}
