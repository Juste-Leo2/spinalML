package spinalML.test

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.nn._
import spinalML.attention.ClassicalAttention
import spinalML.dtypes.{I8, I4, BF16, FP8_E4M3}
import spinalML.examples.HighLevelAttentionTemplate

/**
 * Weight-only quantization (wXaY) through the high-level Sequential API:
 * float activations (BF16/FP8) + SInt weights (I8/I4) declared via
 * customWeightType + weightScales on ClassicalAttention and Linear layers.
 *
 * Note: the wXa4 schemes (FP4 activations) are excluded here: the SoC DMA
 * path is byte-addressed and does not support sub-byte activation dtypes.
 * They remain covered at RTL level by ClassicalAttentionTest/MultiHeadAttentionTest.
 */
class HighLevelAttentionTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  def attentionModel(numHeads: Int, actDt: HardType[Data], wDt: HardType[Data], scales: Seq[Double]): Sequential =
    Sequential(
      globalDataType = actDt,
      inputShape = Seq(4, 4), // [seqLen, embedDim]
      layers = Seq(
        ClassicalAttention(
          embedDim = 4,
          numHeads = numHeads,
          customWeightType = Some(wDt),
          weightScales = scales
        ),
        Flatten(),
        Linear(
          inFeatures = 16,
          outFeatures = 2,
          customWeightType = Some(wDt),
          weightScales = Seq(0.3)
        )
      ),
      axiConfig = axiConfig
    )

  val quantCombos = Seq(
    ("w8a16", () => I8(), () => BF16()),
    ("w4a16", () => I4(), () => BF16()),
    ("w8a8", () => I8(), () => FP8_E4M3()),
    ("w4a8", () => I4(), () => FP8_E4M3())
  )

  for ((combo, wd, ad) <- quantCombos; heads <- Seq(1, 4)) {
    test(s"Sequential attention $combo heads=$heads compilation") {
      SpinalConfig().generateVerilog(attentionModel(heads, ad(), wd(), Seq.fill(4)(0.5)))
    }
  }

  test("Sequential attention per-channel scales compilation") {
    // embedDim = 4 -> four weight columns -> four scales
    SpinalConfig().generateVerilog(attentionModel(2, BF16(), I8(), Seq(0.5, -1.0, 2.0, 0.25)))
  }

  test("HighLevelAttentionTemplate compilation") {
    SpinalConfig().generateVerilog(HighLevelAttentionTemplate(axiConfig))
  }

  test("HighLevelAttentionTemplate SoC runtime simulation") {
    SimConfig.withVerilator.compile(HighLevelAttentionTemplate(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memorySim.start()

      // Image [4, 8] BF16 = 32 elements of 1.0 (0x3C00)
      val imgBase = 0x1000
      for (_ <- 0 until 8) {
        memorySim.memory.writeBigInt(imgBase, BigInt("3C003C003C003C00", 16), 8)
      }

      // Weights region: attention Wq|Wk|Wv|Wo [32, 8] I8 = 256 B,
      // then Linear W [32, 4] I8 = 128 B, then Linear bias [1, 4] BF16 = 8 B
      val weightBase = 0x2000
      for (_ <- 0 until 60) {
        memorySim.memory.writeBigInt(weightBase, BigInt("0101010101010101", 16), 8)
      }

      def writeAxiLite(addr: BigInt, data: BigInt) = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addr
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

      var validCount = 0
      var timeout = 0
      while (validCount < 4 && timeout < 100000) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }

      assert(validCount == 4, s"Expected 4 output transactions, got $validCount")
      println("HighLevelAttentionTemplate SoC simulation successful!")
    }
  }
}
