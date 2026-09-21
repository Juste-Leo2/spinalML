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

  test("P1-1 Conv2D.spillKSlice: knob validation (Linear gabarit, K*K*inC axis)") {
    // K=3, inC=2 -> KFull=18; Ks=6 (multiple of effLanes=9? no: 6 % 9 != 0).
    // Use inC=1, K=3 -> KFull=9, effLanes=9: Ks=9 (P=1) legal.
    val single = Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, spillKSlice = 9)
    assert(single.spilling && single.spillPasses == 1)
    assert(single.spillKFull == 9 && single.spillN == 2)

    // K=2, inC=2 -> KFull=8, effLanes=4: Ks=4, P=2.
    val l = Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 4)
    assert(l.spilling)
    assert(l.spillPasses == 2)

    val legacy = Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2)
    assert(!legacy.spilling)
    assert(legacy.spillPasses == 1)

    intercept[Exception] { Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 3) }
    intercept[Exception] { Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 0) }
    intercept[Exception] { Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 16) }
    // Slice narrower than the internal chunk width is incoherent...
    intercept[Exception] { Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 2) }
    // ...and so is a slice that is not a multiple of narrowed lanes.
    intercept[Exception] {
      Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, weightLanes = 8, spillKSlice = 4)
    }
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

  test("Accelerator: spill needs a descriptor base, fit covers the footprint") {    // No spillBase declared => fail fast, even without capacity.
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

  test("Accelerator S2d: big-K spill exact-fit at scale") {
    // Linear(64 -> 4, Ks=8, P=8) I8: image 64B + weights 264B (256B W @0,
    // 4B bias @256) + out 8B + spill 8B (M*N=4B beat-aligned) = 344B.
    // The on-chip win is the slice fetch: 8*4=32 elems = 4 beats per pass
    // (vs 32 for the full region) — DDR still programs the full W.
    val layers = Seq(Linear(inFeatures = 64, outFeatures = 4, weightLanes = 2, spillKSlice = 8))
    def bigSpillAcc(cap: Option[Long]): Accelerator[Data] = new Accelerator(
      dataType = I8(),
      inputShape = Seq(1, 64),
      modelSpec = layers,
      axiConfig = axiConfig,
      temporal = 1,
      memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = cap)
    )
    SpinalConfig().generateVerilog(bigSpillAcc(Some(344)))
    SpinalConfig().generateVerilog(bigSpillAcc(None))
    intercept[Exception] {
      SpinalConfig().generateVerilog(bigSpillAcc(Some(343)))
    }
    // Slice geometry + spill footprint at scale (node-0 exclusive case).
    val report = SpinalConfig().generateVerilog(spillSequential(layers, inputShape = Seq(1, 64)))
    assert(report.toplevel.totalSpillBytes == 8,
      s"totalSpillBytes=${report.toplevel.totalSpillBytes} != 8")
    assert(report.toplevel.spillSliceInfo.get(0).contains((32, 4)),
      s"spillSliceInfo=${report.toplevel.spillSliceInfo} != (32 elems, 4 beats)")
  }

  test("Sequential S2a: v1 single spilling layer + slice fetch geometry") {
    // Two spilling layers fail fast (single pass controller in v1).
    val double = Seq(
      Linear(inFeatures = 8, outFeatures = 8, weightLanes = 2, spillKSlice = 4),
      Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4))
    intercept[Exception] {
      SpinalConfig().generateVerilog(spillSequential(double))
    }
    // Slice geometry, K=8 N=4 Ks=4 I8 on a 64-bit AXI: slice = 16 elems,
    // 8 elems/beat => 2 beats per pass fetch (vs 4 for the full region).
    val layers = Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4))
    val report = SpinalConfig().generateVerilog(spillSequential(layers))
    assert(report.toplevel.spillLayerIdx.contains(0),
      s"spillLayerIdx=${report.toplevel.spillLayerIdx} should pinpoint layer 0")
    assert(report.toplevel.spillSliceInfo.get(0).contains((16, 2)),
      s"spillSliceInfo=${report.toplevel.spillSliceInfo} != (16 elems, 2 beats)")
    assert(report.toplevel.spillPassIdxOf.contains(0),
      "spilling layer must expose its per-pass index register for S2b")
    // Full-width single pass (P = 1) elaborates: slice == region, 32 elems,
    // 4 beats, and the 1-bit index register stays well-formed.
    val single = Seq(Linear(inFeatures = 8, outFeatures = 4, spillKSlice = 8))
    val reportSingle = SpinalConfig().generateVerilog(spillSequential(single))
    assert(reportSingle.toplevel.spillSliceInfo.get(0).contains((32, 4)),
      s"spillSliceInfo=${reportSingle.toplevel.spillSliceInfo} != (32 elems, 4 beats)")
    // Legacy model: no spill bookkeeping at all.
    val reportPlain = SpinalConfig().generateVerilog(
      spillSequential(Seq(Linear(inFeatures = 8, outFeatures = 4))))
    assert(reportPlain.toplevel.spillLayerIdx.isEmpty)
    assert(reportPlain.toplevel.spillSliceInfo.isEmpty)
  }
}
