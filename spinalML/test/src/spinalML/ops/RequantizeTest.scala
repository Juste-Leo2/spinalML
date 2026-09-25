// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I32, I8}
import spinalML.{RoundingConfig, RoundingMode}
import org.scalatest.funsuite.AnyFunSuite

case class RequantizeTestComp(shift: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I32(), Seq(4), lanes = 4))
    val c = master(Tensor(I8(), Seq(4), lanes = 4))
  }
  
  io.c <> requantize(io.a, I8(), shift)
}

class RequantizeTest extends AnyFunSuite {
  private def runRequantize(shift: Int, inputData: Array[Array[Int]],
                            expectedOutputs: Array[Array[Int]]): Unit = {
    SimConfig.withWave.compile(RequantizeTestComp(shift = shift)).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
      
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

  test("Test streaming Requantize operation I32 -> I8 with shift and saturation") {
    // I32 inputs. The expected outputs follow the elaboration rounding switch
    // (default RNE): -10 >> 2 = -2.5 is a tie, RNE rounds to even (-2) while
    // the legacy truncation lane floors to -3. Both modes are covered
    // bit-exact (see also RoundingPolicyTest).
    val useTrunc = RoundingConfig.current == RoundingMode.Truncate
    val inputData = Array(
      Array(100, -100, 1000, -1000), // shift=2: 25, -25, 250->127, -250->-128
      Array(0, 10, -10, 508)         // shift=2: 0, 2, -2 (RNE) / -3 (trunc), 127
    )
    val expectedOutputs = Array(
      Array(25, -25, 127, -128),
      Array(0, 2, if (useTrunc) -3 else -2, 127)
    )
    runRequantize(shift = 2, inputData, expectedOutputs)
  }

  test("Test Requantize all-dropped shift (RNE always rounds to 0)") {
    // shift >= input width: |value| / 2^shift <= 1/2, so RNE always lands on
    // 0 (the -0.5 tie of Int.MinValue rounds to even). The legacy truncation
    // lane keeps the arithmetic-shift floor: negatives -> -1.
    val useTrunc = RoundingConfig.current == RoundingMode.Truncate
    val inputData = Array(
      Array(0, 1, -1, 127),
      Array(-128, Int.MinValue, Int.MaxValue, -1000)
    )
    val expectedOutputs = inputData.map(_.map(v => if (useTrunc && v < 0) -1 else 0))
    runRequantize(shift = 32, inputData, expectedOutputs)
  }

  test("Test Requantize compilation on I32_I8_shift2") {
    SpinalConfig().generateVerilog(RequantizeTestComp(shift = 2))
  }
}
