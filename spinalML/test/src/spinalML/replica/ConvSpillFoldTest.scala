// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.{BF16, I8}
import spinalML.nn.{Conv2D, LayerSpec}

/**
 * P1-4 — pass-folded replica for spilling Conv2D layers.
 *
 * Same S2 numeric contract as Linear (docs/ddr_spill_ops.md): same logical
 * model => same oracle logits whether the convolution spills or not
 * (full-width partials, single final bias, Ks % effLanes == 0), over the
 * flattened K*K*inChannels axis. No hardware elaborated beyond the dummy
 * layout context.
 */
class ConvSpillFoldTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  def buildLayout(
    layers: Seq[LayerSpec],
    mkDtype: () => Data
  ): WeightMemoryLayout.PackedWeightsResult = {
    var packed: WeightMemoryLayout.PackedWeightsResult = null
    SpinalConfig(targetDirectory = "out/tmp-conv-spill-fold").generateVerilog(new Component {
      setDefinitionName("ConvSpillFoldDummy")
      packed = WeightMemoryLayout.buildDeterministicWeights(layers, mkDtype(), axiConfig)
    })
    packed
  }

  // K=2, inC=2 -> KFull=8, effLanes=4; Ks=4, P=2. H=W=4 -> 3x3 windows.
  val kSize = 2; val inC = 2; val outC = 2; val ks = 4
  val kFull = kSize * kSize * inC
  val shape = Seq(4, 4, inC)
  def spillLayers = Seq(Conv2D(inChannels = inC, outChannels = outC, kernelSize = kSize, spillKSlice = ks))
  def denseLayers = Seq(Conv2D(inChannels = inC, outChannels = outC, kernelSize = kSize))

  test("P1-4: spilling Conv2D I8 emits slice-transposed weights") {
    val info = buildLayout(spillLayers, () => I8()).layers.head
    assert(info.spillKSlice == ks && info.spillPasses == 2)
    val legacy = (0 until kFull * outC).map(idx => ((idx % 7) + 1).toLong)
    val expected = (0 until kFull by ks).flatMap(p =>
      (0 until outC).flatMap(o => (p until p + ks).map(k => legacy(o * kFull + k))))
    assert(info.weightInts == expected,
      s"weightInts not slice-transposed:\n got=${info.weightInts}\n exp=$expected")
  }

  test("P1-4: int spill fold matches dense oracle on the same logical model") {
    val spillPacked = buildLayout(spillLayers, () => I8())
    val densePacked = buildLayout(denseLayers, () => I8())
    val inInts = (0 until shape.product).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), densePacked).logits
    assert(spillOut == denseOut, s"spill $spillOut != dense $denseOut")
  }

  test("P1-4: float spill fold is bit-exact vs dense oracle") {
    val spillPacked = buildLayout(spillLayers, () => BF16())
    val densePacked = buildLayout(denseLayers, () => BF16())
    val inVals = (0 until shape.product).map { idx =>
      val sign = if (idx % 2 == 0) 1.0 else -1.0
      HWArithmetic.fromDouble(sign * (((idx % 7) + 1) * 0.125), 8, 7)
    }
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), densePacked).logits
    val dev = spillOut.zip(denseOut).map { case (a, b) => math.abs(a - b) }.max
    assert(dev == 0.0, s"float conv spill fold deviates: dev=$dev spill=$spillOut dense=$denseOut")
  }

  test("P1-4: handler rejects layer/layout spill mismatch (no silent double transpose)") {
    val spillPacked = buildLayout(spillLayers, () => I8())
    val inInts = (0 until shape.product).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
    assertThrows[IllegalArgumentException] {
      ModelReplica.forwardWithTrace(
        denseLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), spillPacked)
    }
  }
}
