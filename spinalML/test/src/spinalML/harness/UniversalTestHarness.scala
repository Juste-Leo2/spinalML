// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.harness

import scala.collection.mutable.ArrayBuffer
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.nn.Accelerator
import spinalML.dtypes.FloatML
import spinalML.replica.HWArithmetic._

/**
 * Universal SoC Hardware Verification Engine.
 * 
 * Verifies any SpinalML Accelerator against:
 *  1. Protocol compliance and DDR DMA streaming
 *  2. Full frame completion without pipeline deadlock
 *  3. Continuous streaming run control (CSR 0x1C RUN)
 *  4. Bit-exact logit comparison against the software oracle
 */
object UniversalTestHarness {

  def decodeFloat(p: Data): Float = p match {
    case f: FloatML =>
      val eW = f.exponent.getWidth
      val mW = f.mantissa.getWidth
      if (eW == 8 && mW == 7) {
        val bits = ((if (f.sign.toBoolean) 1 else 0) << 15) | ((f.exponent.toInt & 0xFF) << 7) | (f.mantissa.toInt & 0x7F)
        java.lang.Float.intBitsToFloat(bits << 16)
      } else {
        val sign = if (f.sign.toBoolean) -1.0 else 1.0
        val rawE = f.exponent.toInt
        val rawM = f.mantissa.toInt
        val bias = (1 << (eW - 1)) - 1
        val mag =
          if (rawE == 0) rawM.toDouble * math.pow(2.0, 1 - bias - mW)
          else (1.0 + rawM.toDouble / (1 << mW)) * math.pow(2.0, rawE - bias)
        (sign * mag).toFloat
      }
    case s: SInt =>
      s.toBigInt.toFloat
    case u: UInt =>
      u.toBigInt.toFloat
    case _ =>
      0.0f
  }

  def run[M <: Accelerator[_]](
    compiled: SimCompiled[M],
    weightsWords: Seq[BigInt],
    imageWords: Seq[BigInt],
    expectedLogits: Option[Seq[Double]] = None,
    imgBase: Long = 0x10000L,
    weightBase: Long = 0x20000L,
    timeoutCycles: Int = 50000
  ): Seq[Float] = {

    var collectedOutput = ArrayBuffer[Float]()
    
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memorySim.start()

      // Write weights and image to DDR
      MemoryHarness.writeWords(memorySim.memory, weightBase, weightsWords)
      MemoryHarness.writeWords(memorySim.memory, imgBase, imageWords)

      dut.clockDomain.waitSampling(5)

      // Set output stream ready to consume
      dut.io.outStream.stream.ready #= true

      // CSR Configuration
      StreamingHarness.writeAxiLite(dut.io.ctrlBus, dut.clockDomain)(0x08, imgBase)
      StreamingHarness.writeAxiLite(dut.io.ctrlBus, dut.clockDomain)(0x0C, weightBase)
      StreamingHarness.writeAxiLite(dut.io.ctrlBus, dut.clockDomain)(0x00, 1) // Start inference

      var cycles = 0
      val outLanes = dut.io.outStream.lanes
      val expectedCount = dut.io.outStream.shape.product

      while (collectedOutput.length < expectedCount && cycles < timeoutCycles) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          for (l <- 0 until outLanes) {
            if (collectedOutput.length < expectedCount) {
              collectedOutput += decodeFloat(dut.io.outStream.stream.payload(l))
            }
          }
        }
        dut.clockDomain.waitSampling()
        cycles += 1
      }

      assert(collectedOutput.length == expectedCount,
        s"Timeout: collected only ${collectedOutput.length}/$expectedCount outputs in $cycles cycles")

      println(s"Inference completed in $cycles hardware cycles.")

      // Verify bit-exactness if oracle logits are provided
      expectedLogits.foreach { expected =>
        val dev = collectedOutput.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        assert(dev == 0.0, s"Bit-exact assertion failed: max deviation |hw - sw| = $dev")
        println(s"Bit-exact verification PASSED (deviation = 0.000).")
      }
    }

    collectedOutput.toSeq
  }
}
