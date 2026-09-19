// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8
import spinalML.memory.MemoryKind

/**
 * Phase-2 DDR plumbing guards (docs/ddr_impl.md):
 * - `CsrMap` freezes the historical CSR addresses and reserves the
 *   spill/stride/status plane without collision.
 * - `MemorySpec` keeps legacy defaults and validates its inputs.
 * - `Sequential.totalWeightBytes` matches the hand-computed `MemLayout`
 *   footprint (single source of truth, no duplicated layout loop).
 * - `Accelerator` fails fast at elaboration when the footprint exceeds the
 *   declared capacity, and elaborates unchanged otherwise.
 */
class MemorySpecTest extends AnyFunSuite {

  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  test("CsrMap: historical addresses frozen, reserves disjoint") {
    assert(CsrMap.Start == 0x00)
    assert(CsrMap.Status == 0x04)
    assert(CsrMap.ImgBase == 0x08)
    assert(CsrMap.WeightBase == 0x0C)
    assert(CsrMap.Mode == 0x10)
    assert(CsrMap.Reload == 0x14)
    assert(CsrMap.TileCnt == 0x18)
    assert(CsrMap.Run == 0x1C)
    assert(CsrMap.OutAddr == 0x20)
    assert(CsrMap.OutCtrl == 0x24)
    assert(CsrMap.DmaStatus == 0x28)
    assert(CsrMap.DequantScale == 0x30)
    assert(CsrMap.SpillBase == 0x34)
    assert(CsrMap.OutStride == 0x38)
    assert(CsrMap.MemStatus == 0x3C)
    assert(CsrMap.wired.size == 12)
    assert(CsrMap.reserved.size == 3)
    assert((CsrMap.wired & CsrMap.reserved).isEmpty, "reserved CSR collides with a wired address")
  }

  test("MemorySpec: legacy defaults and input validation") {
    val d = MemorySpec.default
    assert(d.kind == MemoryKind.OnChip)
    assert(d.imgBase == 0x10000L)
    assert(d.weightBase == 0x20000L)
    assert(d.spillBase.isEmpty)
    assert(d.capacityBytes.isEmpty)

    intercept[Exception] { MemorySpec(imgBase = -1) }
    intercept[Exception] { MemorySpec(weightBase = -8) }
    intercept[Exception] { MemorySpec(spillBase = Some(-1)) }
    intercept[Exception] { MemorySpec(capacityBytes = Some(0)) }
  }

  test("MemorySpec.reportFit: exact boundary accounting") {
    // Pure-Scala check: 4B image + 16B weights + 8B out = 28B.
    MemorySpec(capacityBytes = Some(28)).reportFit(4, 16, 8)
    intercept[Exception] {
      MemorySpec(capacityBytes = Some(27)).reportFit(4, 16, 8)
    }
    // No capacity declared => no check (legacy behavior).
    MemorySpec.default.reportFit(Long.MaxValue / 2, 0, 0)
  }

  test("Sequential.totalWeightBytes matches the MemLayout footprint") {
    // Linear(4 -> 2) in I8 on a 64-bit beat (8B):
    // weights 8 elems => 8B @0, then align(8) = 8;
    // bias 2 elems => 2B @8, then align(10) = 16. Total = 16B.
    val report = SpinalConfig().generateVerilog(Sequential(
      globalDataType = I8(),
      inputShape = Seq(1, 4),
      layers = Seq(Linear(inFeatures = 4, outFeatures = 2)),
      axiConfig = axiConfig
    ))
    assert(report.toplevel.totalWeightBytes == 16,
      s"totalWeightBytes=${report.toplevel.totalWeightBytes} != 16")
  }

  private def tinyAccelerator(memory: MemorySpec): Accelerator[Data] = new Accelerator(
    dataType = I8(),
    inputShape = Seq(1, 4),
    modelSpec = Seq(Linear(inFeatures = 4, outFeatures = 2)),
    axiConfig = axiConfig,
    memory = memory
  )

  test("Accelerator: elaborates with fitting capacity, rejects overflow") {
    // Footprint of the toy model: image 4B + weights 16B + out 8B = 28B.
    SpinalConfig().generateVerilog(tinyAccelerator(MemorySpec(capacityBytes = Some(28))))
    SpinalConfig().generateVerilog(tinyAccelerator(MemorySpec.default))
    intercept[Exception] {
      SpinalConfig().generateVerilog(tinyAccelerator(MemorySpec(capacityBytes = Some(27))))
    }
    intercept[Exception] {
      SpinalConfig().generateVerilog(tinyAccelerator(MemorySpec(capacityBytes = Some(1))))
    }
  }
}
