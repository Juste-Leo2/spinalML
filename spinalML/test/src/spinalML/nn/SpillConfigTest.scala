// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8

/**
 * S0 compute-side spill guards (docs/ddr_final_impl.md): elaboration-time
 * knobs and requires only — no RTL behavior change.
 * - `Linear.spillKSlice` validates like `weightLanes` (divisor of inFeatures,
 *   multiple of effLanes so the pass-internal chunking matches the replica).
 * - `Sequential` accepts a spilling layer only with `temporal >= 1`
 *   (windowed row drain), in STREAM_PER_PASS (no residency), with a
 *   re-streamable A (node 0 DDR-backed, or within `spillReplayBudgetBytes`),
 *   and exposes the exact `totalSpillBytes` footprint.
 * - `Accelerator` fit-checks the computed spill footprint and fails fast
 *   when no spill descriptor base is declared.
 */
class SpillConfigTest extends AnyFunSuite {

  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  test("Linear.spillKSlice: knob validation") {
    // Slice 4 of K=8 with narrow lanes 2: 4 % 2 == 0, P = 2 passes.
    val l = Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)
    assert(l.spilling)
    assert(l.spillPasses == 2)

    val legacy = Linear(inFeatures = 8, outFeatures = 4)
    assert(!legacy.spilling)
    assert(legacy.spillPasses == 1)

    intercept[Exception] { Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 3) }
    intercept[Exception] { Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 0) }
    intercept[Exception] { Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 16) }
    // A slice narrower than the internal chunk width is incoherent...
    intercept[Exception] { Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 4) }
    // ...and so is a slice that is not a multiple of narrowed lanes.
    intercept[Exception] {
      Linear(inFeatures = 8, outFeatures = 4, weightLanes = 4, spillKSlice = 2)
    }
    // Full-width single pass (P = 1) stays legal.
    val single = Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 8)
    assert(single.spilling && single.spillPasses == 1)
  }

  private def spillSequential(
    layers: Seq[LayerSpec],
    inputShape: Seq[Int] = Seq(1, 8),
    temporal: Int = 1,
    weightResidency: Boolean = false,
    spillReplayBudgetBytes: Int = 4096
  ): Sequential = Sequential(
    globalDataType = I8(),
    inputShape = inputShape,
    layers = layers,
    axiConfig = axiConfig,
    weightResidency = weightResidency,
    temporal = temporal,
    spillReplayBudgetBytes = spillReplayBudgetBytes
  )

  test("Sequential: spill requires temporal >= 1") {
    // The windowed row drain is the spill drain path (S1/S2).
    val layers = Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4))
    intercept[Exception] {
      SpinalConfig().generateVerilog(spillSequential(layers, temporal = 0))
    }
    // NOTE: no elaboration require on weightResidency — the flag only wires
    // the (default-off) control plane. The STREAM_PER_PASS runtime contract
    // is enforced by the S2 pass controller.
    SpinalConfig().generateVerilog(spillSequential(layers, weightResidency = true))
  }

  test("Sequential: node-0 spill elaborates and sizes totalSpillBytes") {
    // M=1, N=4, I8 acc: 4B region => beat-aligned (8B) = 8.
    val layers = Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4))
    val report = SpinalConfig().generateVerilog(spillSequential(layers))
    assert(report.toplevel.totalSpillBytes == 8,
      s"totalSpillBytes=${report.toplevel.totalSpillBytes} != 8")
    // DDR-backed node 0 ignores the replay budget, even at zero.
    val reportNoReplay = SpinalConfig().generateVerilog(
      spillSequential(layers, spillReplayBudgetBytes = 0))
    assert(reportNoReplay.toplevel.totalSpillBytes == 8)
    // No spilling layer => zero footprint, legacy behavior.
    val reportPlain = SpinalConfig().generateVerilog(
      spillSequential(Seq(Linear(inFeatures = 8, outFeatures = 4))))
    assert(reportPlain.toplevel.totalSpillBytes == 0)
  }

  test("Sequential: deep spill needs the replay budget") {
    val deep = Seq(
      Linear(inFeatures = 8, outFeatures = 8),
      Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4))
    // Layer-1 A = 8B I8, within the default 4096B budget (replay path).
    val report = SpinalConfig().generateVerilog(spillSequential(deep))
    // Layer 0: nothing. Layer 1: M=1, N=4 I8 => 4B => 8B aligned.
    assert(report.toplevel.totalSpillBytes == 8,
      s"totalSpillBytes=${report.toplevel.totalSpillBytes} != 8")
    // ...but not with the replay path disabled.
    intercept[Exception] {
      SpinalConfig().generateVerilog(spillSequential(deep, spillReplayBudgetBytes = 0))
    }
    intercept[Exception] {
      SpinalConfig().generateVerilog(Sequential(
        globalDataType = I8(),
        inputShape = Seq(1, 8),
        layers = deep,
        axiConfig = axiConfig,
        temporal = 1,
        spillReplayBudgetBytes = -1))
    }
  }

  private def spillAccelerator(memory: MemorySpec): Accelerator[Data] = new Accelerator(
    dataType = I8(),
    inputShape = Seq(1, 8),
    modelSpec = Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
    axiConfig = axiConfig,
    temporal = 1,
    memory = memory
  )

  test("Accelerator: spill needs a descriptor base, fit covers the footprint") {
    // No spillBase declared => fail fast, even without capacity.
    intercept[Exception] {
      SpinalConfig().generateVerilog(spillAccelerator(MemorySpec.default))
    }
    // Footprint: image 8B + weights 40B + out 8B + spill 8B = 64B.
    SpinalConfig().generateVerilog(spillAccelerator(
      MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(64))))
    SpinalConfig().generateVerilog(spillAccelerator(
      MemorySpec(spillBase = Some(0x30000L))))
    intercept[Exception] {
      SpinalConfig().generateVerilog(spillAccelerator(
        MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(63))))
    }
  }
}
