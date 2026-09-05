// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.{FloatML, BF16, I8}
import spinalML.examples.ResidualMLPTemplate

/**
 * DAG topology through the high-level API: explicit Add / Concat merge nodes
 * referencing earlier graph nodes, with automatic fan-out taps.
 *
 * The residual golden values are ReLU(X + 0.5 * ReLU(X)) with an identity
 * first dense layer, verified against the Python goldens (BF16-exact).
 */
class DagTopologyTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF

  def word(elems: Seq[Int]): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0))((acc, e) => acc | (BigInt(e._1 & 0xFFFF) << (16 * e._2)))

  def getFloat(p: Data): Float = {
    val f = p.asInstanceOf[FloatML]
    val bits = ((if (f.sign.toBoolean) 1 else 0) << 15) | ((f.exponent.toInt & 0xFF) << 7) | (f.mantissa.toInt & 0x7F)
    java.lang.Float.intBitsToFloat(bits << 16)
  }

  test("ResidualMLPTemplate compilation") {
    SpinalConfig().generateVerilog(ResidualMLPTemplate(axiConfig))
  }

  test("Concat DAG compilation") {
    SpinalConfig().generateVerilog(
      Sequential(
        globalDataType = BF16(),
        inputShape = Seq(2, 4),
        layers = Seq(
          Linear(inFeatures = 4, outFeatures = 4),
          ReLU(),
          Linear(inFeatures = 4, outFeatures = 4),
          Concat(a = 0, b = 3)
        ),
        axiConfig = axiConfig
      )
    )
  }

  test("DAG validation: forward reference rejected") {
    intercept[Exception] {
      Sequential(
        globalDataType = BF16(),
        inputShape = Seq(2, 4),
        layers = Seq(
          Linear(inFeatures = 4, outFeatures = 4),
          Add(a = 2, b = 1)
        ),
        axiConfig = axiConfig
      )
    }
  }

  test("DAG validation: dtype mismatch rejected") {
    intercept[Exception] {
      Sequential(
        globalDataType = BF16(),
        inputShape = Seq(2, 4),
        layers = Seq(
          Cast(targetType = I8()),
          Linear(inFeatures = 4, outFeatures = 4),
          Add(a = 0, b = 2)
        ),
        axiConfig = axiConfig
      )
    }
  }

  test("DAG validation: shape mismatch on Add rejected") {
    intercept[Exception] {
      Sequential(
        globalDataType = BF16(),
        inputShape = Seq(2, 4),
        layers = Seq(
          Linear(inFeatures = 4, outFeatures = 5),
          Add(a = 0, b = 1)
        ),
        axiConfig = axiConfig
      )
    }
  }

  test("Residual MLP SoC runtime golden (skip connection, BF16 [2,4])") {
    SimConfig.withVerilator.compile(ResidualMLPTemplate(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memorySim.start()

      // X = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.25, -2.0]]
      val imgBase = 0x1000
      val xWords = Seq(
        Seq(1.0f, 2.0f, 3.0f, 4.0f),
        Seq(-1.0f, 0.5f, 0.25f, -2.0f)
      ).map(row => word(row.map(bf16Bits)))
      for ((w, i) <- xWords.zipWithIndex) memorySim.memory.writeBigInt(imgBase + i * 8, w, 8)

      // Weights region in declaration order:
      // Dense1 W [4][4] identity, Dense1 b zeros,
      // Dense2 W [4][4] = 0.5 * identity, Dense2 b zeros
      // (stored [outFeatures, inFeatures], one row per AXI beat)
      val weightBase = 0x2000
      val zero = bf16Bits(0.0f)
      val half = bf16Bits(0.5f)
      var addr = weightBase
      def put(wordElems: Seq[Int]): Unit = {
        memorySim.memory.writeBigInt(addr, word(wordElems), 8)
        addr += 8
      }
      for (n <- 0 until 4) put(Seq.tabulate(4)(k => if (k == n) bf16Bits(1.0f) else zero))
      put(Seq(zero, zero, zero, zero))
      for (n <- 0 until 4) put(Seq.tabulate(4)(k => if (k == n) half else zero))
      put(Seq(zero, zero, zero, zero))

      def writeAxiLite(addrL: BigInt, data: BigInt) = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addrL
        dut.io.ctrlBus.w.valid #= true
        dut.io.ctrlBus.w.payload.data #= data
        dut.io.ctrlBus.w.payload.strb #= 0xF
        dut.io.ctrlBus.b.ready #= true

        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
        dut.io.ctrlBus.b.ready #= false
        dut.clockDomain.waitSampling()
      }

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.outStream.stream.ready #= true

      dut.clockDomain.waitSampling(5)

      writeAxiLite(0x08, imgBase)
      writeAxiLite(0x0C, weightBase)
      writeAxiLite(0x00, 1)

      val expected = Seq(1.5f, 3.0f, 4.5f, 6.0f, 0.0f, 0.75f, 0.375f, 0.0f)
      val collected = scala.collection.mutable.ArrayBuffer[Float]()
      var timeout = 0
      while (collected.length < expected.length && timeout < 100000) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          collected += getFloat(dut.io.outStream.stream.payload(0))
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }

      assert(timeout < 100000, s"Timeout: collected ${collected.length}/${expected.length}")
      assert(collected.toSeq == expected, s"Expected $expected, got $collected")
    }
  }
}
