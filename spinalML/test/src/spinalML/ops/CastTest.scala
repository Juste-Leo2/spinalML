package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.{BF16, FloatML}
import spinalML.dtypes.BF16Sim
import org.scalatest.funsuite.AnyFunSuite

case class CastTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(SInt(8 bits), Seq(4), lanes = 4))
    val c = master(Tensor(BF16(), Seq(4), lanes = 4))
  }
  
  val casted = cast(io.a, BF16())
  io.c <> casted
}

class CastTest extends AnyFunSuite {
  test("Test streaming Cast operation SInt -> BF16") {
    SimConfig.withWave.compile(CastTestComp()).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
      val inputData = Array(
        Array(5, -5, 0, 127),
        Array(-128, 1, -1, 42)
      )
      
      var outputIndex = 0
      val numExpectedOutputs = inputData.length
      
      // Monitor output
      fork {
        while (outputIndex < numExpectedOutputs) {
          dut.clockDomain.waitSampling()
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            
            val expectedInputs = inputData(outputIndex)
            for (i <- 0 until 4) {
              val signC = if(dut.io.c.stream.payload(i).asInstanceOf[FloatML].sign.toBoolean) 1 else 0
              val expC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].exponent.toInt
              val mantC = dut.io.c.stream.payload(i).asInstanceOf[FloatML].mantissa.toInt
              val bitsC = (signC << 15) | (expC << 7) | mantC
              
              val fC = BF16Sim.bf16BitsToFloat(bitsC)
              val expected = expectedInputs(i).toFloat
              
              val expectedBf16Bits = BF16Sim.floatToBf16Bits(expected)
              val expectedBf16Float = BF16Sim.bf16BitsToFloat(expectedBf16Bits)
              
              assert(fC == expectedBf16Float, s"Index $outputIndex Lane $i: Expected $expectedBf16Float, got $fC")
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
        // Random wait between beats
        dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
      }
      
      // Wait for all outputs to be checked
      dut.clockDomain.waitSamplingWhere(outputIndex == numExpectedOutputs)
      dut.clockDomain.waitSampling(5)
    }
  }
}
