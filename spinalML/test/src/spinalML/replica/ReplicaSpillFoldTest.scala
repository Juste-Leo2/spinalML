// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.{BF16, I8}
import spinalML.nn.{LayerSpec, Linear}
import spinalML.replica.HWArithmetic._

/**
 * R2 — pass-folded replica for spilling Linear layers. Same logical model
 * => same oracle logits whether the layer spills or not (full-width
 * partials, single final bias, Ks % effLanes == 0), int (associative) and
 * float (bit-exact fadd order) paths. No hardware beyond the dummy context.
 */
class ReplicaSpillFoldTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  def buildLayout(
    layers: Seq[LayerSpec],
    mkDtype: () => Data
  ): WeightMemoryLayout.PackedWeightsResult = {
    var packed: WeightMemoryLayout.PackedWeightsResult = null
    SpinalConfig(targetDirectory = "out/tmp-replica-spill-fold").generateVerilog(new Component {
      setDefinitionName("ReplicaSpillFoldDummy")
      packed = WeightMemoryLayout.buildDeterministicWeights(layers, mkDtype(), axiConfig)
    })
    packed
  }

  test("R2: int spill fold matches dense oracle on the same logical model") {
    val k = 8; val n = 4; val ks = 4
    val spillLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks))
    val denseLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2))
    val spillPacked = buildLayout(spillLayers, () => I8())
    val densePacked = buildLayout(denseLayers, () => I8())

    val shape = Seq(1, k)
    val inInts = (0 until k).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), densePacked).logits
    assert(spillOut == denseOut, s"spill $spillOut != dense $denseOut")
  }

  test("R2: float spill fold is bit-exact vs dense oracle") {
    val k = 8; val n = 4; val ks = 4
    val spillLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks))
    val denseLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2))
    val spillPacked = buildLayout(spillLayers, () => BF16())
    val densePacked = buildLayout(denseLayers, () => BF16())

    val shape = Seq(1, k)
    val inVals = (0 until k).map { idx =>
      val sign = if (idx % 2 == 0) 1.0 else -1.0
      fromDouble(sign * (((idx % 7) + 1) * 0.125), 8, 7)
    }
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), densePacked).logits
    val dev = spillOut.zip(denseOut).map { case (a, b) => math.abs(a - b) }.max
    assert(dev == 0.0, s"float spill fold deviates: dev=$dev spill=$spillOut dense=$denseOut")
  }

  test("R2: linear spillKSlice=K matches the legacy single-pass path") {
    val k = 8; val n = 4
    val packed = buildLayout(Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2)), () => BF16())
    val w = packed.layers.head
    val fcW = (0 until n).map(o => w.weightValues.slice(o * k, (o + 1) * k))
    val fcB = (0 until n).map(o => w.biasValues(o))
    val inVals = (0 until k).map(idx => fromDouble((((idx % 7) + 1) * 0.0625), 8, 7))
    val legacy = LayerReplicas.linear(inVals, fcW, fcB, 8, 7, 2)
    val folded = LayerReplicas.linear(inVals, fcW, fcB, 8, 7, 2, spillKSlice = k)
    assert(legacy.zip(folded).forall { case (a, b) => decode(a, 8, 7) == decode(b, 8, 7) },
      "spillKSlice=K must reproduce the legacy path exactly")
  }

  test("R2: handler rejects layer/layout spill mismatch (no silent double transpose)") {    val k = 8; val n = 4
    val spillPacked = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = 4)), () => I8())
    val denseLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2))
    val shape = Seq(1, k)
    val inInts = (0 until k).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
    assertThrows[IllegalArgumentException] {
      ModelReplica.forwardWithTrace(
        denseLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), spillPacked)
    }
  }

  test("P0a: int spill fold matches dense oracle with M=2 rows") {
    val m = 2; val k = 8; val n = 4; val ks = 4
    val spillLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks))
    val denseLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2))
    val spillPacked = buildLayout(spillLayers, () => I8())
    val densePacked = buildLayout(denseLayers, () => I8())

    val shape = Seq(m, k)
    val inInts = (0 until m * k).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.IntTensor(shape, inInts, 8), densePacked).logits
    assert(spillOut.length == m * n, s"spill logits ${spillOut.length} != M*N=${m * n}")
    assert(spillOut == denseOut, s"spill $spillOut != dense $denseOut")
  }

  test("P0a: float spill fold is bit-exact vs dense oracle with M=2 rows") {
    val m = 2; val k = 8; val n = 4; val ks = 4
    val spillLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks))
    val denseLayers = Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2))
    val spillPacked = buildLayout(spillLayers, () => BF16())
    val densePacked = buildLayout(denseLayers, () => BF16())

    val shape = Seq(m, k)
    val inVals = (0 until m * k).map { idx =>
      val sign = if (idx % 2 == 0) 1.0 else -1.0
      fromDouble(sign * (((idx % 7) + 1) * 0.125), 8, 7)
    }
    val spillOut = ModelReplica.forwardWithTrace(
      spillLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), spillPacked).logits
    val denseOut = ModelReplica.forwardWithTrace(
      denseLayers, shape, ModelReplica.FloatTensor(shape, inVals, 8, 7), densePacked).logits
    val dev = spillOut.zip(denseOut).map { case (a, b) => math.abs(a - b) }.max
    assert(dev == 0.0, s"float M=2 spill fold deviates: dev=$dev spill=$spillOut dense=$denseOut")
  }
}
