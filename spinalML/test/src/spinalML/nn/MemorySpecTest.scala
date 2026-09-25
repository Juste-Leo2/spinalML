// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8
import spinalML.memory.MemoryKind

/**
 * DDR plumbing guards:
 * - `CsrMap` freezes the CSR addresses (spill base live, stride/status
 *   still reserved) without collision.
 * - `MemorySpec` keeps legacy defaults and validates its inputs.
 * - `Sequential.totalWeightBytes` matches the hand-computed `MemLayout`
 *   footprint (single source of truth, no duplicated layout loop).
 * - `Accelerator` fails fast at elaboration when the footprint exceeds the
 *   declared capacity, and elaborates unchanged otherwise.
 * - CSR 0x34 (spill base) reads back its descriptor default and host writes.
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
    assert(CsrMap.wired.size == 13)
    assert(CsrMap.reserved.size == 2)
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
    // Pure-Scala check: 4B image + 16B weights + 8B out + 0B spill = 28B.
    MemorySpec(capacityBytes = Some(28)).reportFit(4, 16, 8)
    intercept[Exception] {
      MemorySpec(capacityBytes = Some(27)).reportFit(4, 16, 8)
    }
    // Spill footprint participates in the same boundary.
    MemorySpec(capacityBytes = Some(36), spillBytes = Some(8)).reportFit(4, 16, 8, 8)
    intercept[Exception] {
      MemorySpec(capacityBytes = Some(35), spillBytes = Some(8)).reportFit(4, 16, 8, 8)
    }
    intercept[Exception] { MemorySpec(spillBytes = Some(-1)) }
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

  test("Accelerator S2d: big-K legacy model fails fast under a small capacity") {
    // Linear(64 -> 4) I8, no spill: image 64B + weights 264B (256B W @0,
    // 4B bias @256) + out 8B = 336B. Under a 128B descriptor the
    // elaboration fail-fast must bite at real sizes, not just toy ones.
    def bigAcc(memory: MemorySpec): Accelerator[Data] = new Accelerator(
      dataType = I8(),
      inputShape = Seq(1, 64),
      modelSpec = Seq(Linear(inFeatures = 64, outFeatures = 4)),
      axiConfig = axiConfig,
      memory = memory
    )
    intercept[Exception] {
      SpinalConfig().generateVerilog(bigAcc(MemorySpec(capacityBytes = Some(128))))
    }
    // ...and the same model elaborates once the capacity covers the real
    // footprint (the gate is exact, not spill-specific).
    SpinalConfig().generateVerilog(bigAcc(MemorySpec(capacityBytes = Some(336))))
  }

  test("Accelerator: CSR spill base (0x34) defaults and readback") {
    val spinalConfig = SpinalConfig()
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      tinyAccelerator(MemorySpec(spillBase = Some(0x30000L)))
    )
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      def writeCsr(addr: BigInt, data: BigInt): Unit = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addr
        dut.io.ctrlBus.w.valid #= true
        dut.io.ctrlBus.w.payload.data #= data
        dut.io.ctrlBus.w.payload.strb #= 0xF
        dut.io.ctrlBus.b.ready #= true
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSampling()
      }
      def readCsr(addr: BigInt): BigInt = {
        dut.io.ctrlBus.ar.valid #= true
        dut.io.ctrlBus.ar.payload.addr #= addr
        dut.io.ctrlBus.r.ready #= true
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.ar.ready.toBoolean)
        dut.io.ctrlBus.ar.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.r.valid.toBoolean)
        val data = dut.io.ctrlBus.r.payload.data.toBigInt
        dut.io.ctrlBus.r.ready #= false
        dut.clockDomain.waitSampling()
        data
      }

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= true
      dut.io.ctrlBus.r.ready #= false
      dut.clockDomain.waitSampling(5)

      // Reset value comes from the MemorySpec descriptor...
      assert(readCsr(CsrMap.SpillBase) == 0x30000L,
        s"CSR 0x34 reset value must be the descriptor spillBase")
      // ...and host writes stick.
      writeCsr(CsrMap.SpillBase, 0x40000L)
      assert(readCsr(CsrMap.SpillBase) == 0x40000L,
        s"CSR 0x34 must read back the host-programmed spill base")
      // Neighboring registers are unaffected (no address aliasing).
      assert(readCsr(CsrMap.DequantScale) == 0,
        s"CSR 0x30 must be untouched by 0x34 writes (no descriptor Cast here)")
    }
  }
}
