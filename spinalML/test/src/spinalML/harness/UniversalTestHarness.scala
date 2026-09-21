// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.harness

import scala.collection.mutable.ArrayBuffer
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinal.lib.bus.amba4.axilite.AxiLite4
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
    timeoutCycles: Int = 50000,
    // Seam for `--stress` (timing/pressure only, the bit-exact oracle stays
    // untouched): stress supplies a degraded memory-model config here.
    // Default = the historical ideal model.
    memorySimConfig: AxiMemorySimConfig = AxiMemorySimConfig(maxOutstandingReads = 8)
  ): Seq[Float] = {
    var result: Seq[Float] = null
    compiled.doSim { dut =>
      result = execute(
        cd = dut.clockDomain,
        axi = dut.io.axiMaster,
        ctrl = dut.io.ctrlBus,
        outShape = dut.io.outStream.shape,
        outLanes = dut.io.outStream.lanes,
        outValid = () => dut.io.outStream.stream.valid.toBoolean,
        setOutReady = (v: Boolean) => dut.io.outStream.stream.ready #= v,
        outPayload = (l: Int) => dut.io.outStream.stream.payload(l),
        weightsWords = weightsWords,
        imageWords = imageWords,
        expectedLogits = expectedLogits,
        imgBase = imgBase,
        weightBase = weightBase,
        timeoutCycles = timeoutCycles,
        memorySimConfig = memorySimConfig,
        tag = "ideal"
      )
    }
    result
  }

  /**
   * Stress entry point (T1, `--stress` CLI flag): same bit-exact engine as
   * `run`, driven through a `ChaosDut` stress top (DUT + `DramChaosInterposer`
   * + exposed ports, elaborated by the caller/scaffold). The oracle path is
   * shared verbatim — only the port accessors differ.
   */
  def runStress[T <: Component with ChaosDut](
    compiled: SimCompiled[T],
    weightsWords: Seq[BigInt],
    imageWords: Seq[BigInt],
    expectedLogits: Option[Seq[Double]] = None,
    imgBase: Long = 0x10000L,
    weightBase: Long = 0x20000L,
    timeoutCycles: Int = 200000,
    memorySimConfig: AxiMemorySimConfig = AxiMemorySimConfig(maxOutstandingReads = 8),
    tag: String = "stress"
  ): Seq[Float] = {
    var result: Seq[Float] = null
    compiled.doSim { top =>
      result = execute(
        cd = top.clockDomain,
        axi = top.memAxi,
        ctrl = top.ctrlAxi,
        outShape = top.outShape,
        outLanes = top.outLanes,
        outValid = () => top.outValid,
        setOutReady = (v: Boolean) => top.setOutReady(v),
        outPayload = (l: Int) => top.outPayload(l),
        weightsWords = weightsWords,
        imageWords = imageWords,
        expectedLogits = expectedLogits,
        imgBase = imgBase,
        weightBase = weightBase,
        timeoutCycles = timeoutCycles,
        memorySimConfig = memorySimConfig,
        tag = tag
      )
    }
    result
  }

  private def execute(
    cd: ClockDomain,
    axi: Axi4,
    ctrl: AxiLite4,
    outShape: Seq[Int],
    outLanes: Int,
    outValid: () => Boolean,
    setOutReady: Boolean => Unit,
    outPayload: Int => Data,
    weightsWords: Seq[BigInt],
    imageWords: Seq[BigInt],
    expectedLogits: Option[Seq[Double]],
    imgBase: Long,
    weightBase: Long,
    timeoutCycles: Int,
    memorySimConfig: AxiMemorySimConfig,
    tag: String
  ): Seq[Float] = {
    var collectedOutput = ArrayBuffer[Float]()
    cd.forkStimulus(period = 10)

    val memorySim = AxiMemorySim(
      axi = axi,
      clockDomain = cd,
      config = memorySimConfig
    )
    memorySim.start()

    // Write weights and image to DDR
    MemoryHarness.writeWords(memorySim.memory, weightBase, weightsWords)
    MemoryHarness.writeWords(memorySim.memory, imgBase, imageWords)

    cd.waitSampling(5)

    // Set output stream ready to consume
    setOutReady(true)

    // CSR Configuration
    StreamingHarness.writeAxiLite(ctrl, cd)(0x08, imgBase)
    StreamingHarness.writeAxiLite(ctrl, cd)(0x0C, weightBase)
    StreamingHarness.writeAxiLite(ctrl, cd)(0x00, 1) // Start inference

    var cycles = 0
    val expectedCount = outShape.product

    while (collectedOutput.length < expectedCount && cycles < timeoutCycles) {
      // NOTE: ready is held high, so valid alone gates collection (same as run).
      if (outValid()) {
        for (l <- 0 until outLanes) {
          if (collectedOutput.length < expectedCount) {
            collectedOutput += decodeFloat(outPayload(l))
          }
        }
      }
      cd.waitSampling()
      cycles += 1
    }

    assert(collectedOutput.length == expectedCount,
      s"[$tag] Timeout: collected only ${collectedOutput.length}/$expectedCount outputs in $cycles cycles")

    println(s"[$tag] Inference completed in $cycles hardware cycles.")

    // Verify bit-exactness if oracle logits are provided
    expectedLogits.foreach { expected =>
      val pairs = collectedOutput.toSeq.zip(expected)
      val devs = pairs.map { case (h, s) => math.abs(h.toDouble - s) }
      devs.zipWithIndex.foreach { case (d, i) =>
        if (d != 0.0) {
          val hwVal = pairs(i)._1
          val swVal = pairs(i)._2
          println(f"[$tag DEV] out[$i] hw=$hwVal%9.5f sw=$swVal%9.5f dev=$d%9.6f")
        }
      }
      val dev = devs.max
      assert(dev == 0.0, s"[$tag] Bit-exact assertion failed: max deviation |hw - sw| = $dev")
      println(s"[$tag] Bit-exact verification PASSED (deviation = 0.000).")
    }

    collectedOutput.toSeq
  }
}
