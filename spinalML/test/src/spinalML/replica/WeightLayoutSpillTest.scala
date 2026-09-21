// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.replica

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.{BF16, I8}
import spinalML.nn.{LayerSpec, Linear}

/**
 * R1 — slice-transposed weight layout for spilling Linear layers.
 *
 * Independent reference: the `programmedW(ks)` helper semantics from
 * `SequentialSpillTest` (per-pass contiguous slices `p*Ks*N + n*Ks+k`).
 * Verifies the S2 layout contract (docs/ddr_final_impl.md).
 *
 * Note: building deterministic weights instantiates Spinal `Data` (dtypes),
 * so each case elaborates inside a dummy component context — no hardware is
 * generated or simulated, only the layout tool runs.
 */
class WeightLayoutSpillTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  /** Runs the layout tool inside a dummy elaboration context. */
  def buildLayout(
    layers: Seq[LayerSpec],
    mkDtype: () => Data
  ): WeightMemoryLayout.PackedWeightsResult = {
    var packed: WeightMemoryLayout.PackedWeightsResult = null
    SpinalConfig(targetDirectory = "out/tmp-weight-layout-spill").generateVerilog(new Component {
      setDefinitionName("WeightLayoutSpillDummy")
      packed = WeightMemoryLayout.buildDeterministicWeights(layers, mkDtype(), axiConfig)
    })
    packed
  }

  /** Legacy whole-transpose flat pattern, as emitted for non-spill layers. */
  def legacyPattern(k: Int, n: Int, mod: Int): Seq[Long] =
    (0 until k * n).map(idx => ((idx % mod) + 1).toLong)

  /** Independent slice-transpose reference: pass-major, then output, then local k. */
  def programmedRef(k: Int, n: Int, ks: Int, legacy: Seq[Long]): Seq[Long] =
    (0 until k by ks).flatMap(p => (0 until n).flatMap(o => (p until p + ks).map(c => legacy(o * k + c))))

  test("R1: spilling Linear I8 emits slice-transposed weights, bias untouched") {
    val k = 4; val n = 4; val ks = 2
    val info = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks)),
      () => I8()).layers.head
    assert(info.spillKSlice == ks, s"spillKSlice=${info.spillKSlice} (expected $ks)")
    assert(info.spillPasses == k / ks, s"spillPasses=${info.spillPasses} (expected ${k / ks})")

    val legacy = legacyPattern(k, n, mod = 7)
    assert(info.weightInts == programmedRef(k, n, ks, legacy),
      s"weightInts not slice-transposed:\n got=${info.weightInts}\n exp=${programmedRef(k, n, ks, legacy)}")

    // Bias region keeps the legacy order (fetched once, final pass only).
    val legacyB = (0 until n).map(idx => ((idx % 5) + 1).toLong)
    assert(info.biasInts == legacyB, s"biasInts=${info.biasInts} (expected $legacyB)")
  }

  test("R1: P=1 spill degenerates to the legacy whole-transpose order") {
    val k = 4; val n = 4
    val spill = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = k)),
      () => I8()).layers.head
    val dense = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2)),
      () => I8()).layers.head
    assert(spill.weightInts == dense.weightInts, "P=1 spill must equal the dense layout")
    assert(spill.biasInts == dense.biasInts, "P=1 spill bias must equal the dense bias")
    assert(dense.spillKSlice == -1, s"dense spillKSlice=${dense.spillKSlice} (expected -1)")
  }

  test("R1: non-spilling Linear keeps the legacy layout bit-identical") {
    val k = 8; val n = 4
    val info = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2)),
      () => I8()).layers.head
    assert(info.spillKSlice == -1)
    assert(info.spillPasses == 1)
    assert(info.weightInts == legacyPattern(k, n, mod = 7))
  }

  test("R1: spilling Linear BF16 permutes values, lengths preserved") {
    val k = 4; val n = 2; val ks = 2
    val spill = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2, spillKSlice = ks)),
      () => BF16()).layers.head
    val dense = buildLayout(
      Seq(Linear(inFeatures = k, outFeatures = n, weightLanes = 2)),
      () => BF16()).layers.head
    assert(spill.weightValues.length == k * n)
    assert(spill.weightInts.isEmpty, "float path must leave weightInts empty")
    // Same multiset as dense (pure permutation), different order (transposed).
    def multiset(v: Seq[HWArithmetic.F]) = v.map(f => HWArithmetic.decode(f, 8, 7)).sorted
    assert(multiset(spill.weightValues) == multiset(dense.weightValues),
      "spill values must be a permutation of the dense values")
    assert(spill.weightValues != dense.weightValues, "Ks=2 of K=4 must reorder the region")
  }
}
