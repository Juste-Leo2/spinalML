// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.{PdkFamily, Target}
import spinalML.dtypes.I8
import spinalML.nn.Accelerator
import spinalML.io.UartSoC

class MemoryAdapterTest extends AnyFunSuite {

  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  test("MemoryAdapter: factory resolution from Target") {
    val asicFactory = MemoryAdapter.factory(Target.ASIC(PdkFamily.Sky130), memoryWords = 512)
    SpinalConfig().generateVerilog(asicFactory(axiConfig))

    val fpgaFactory = MemoryAdapter.factory(Target.FPGA(), memoryWords = 512)
    SpinalConfig().generateVerilog(fpgaFactory(axiConfig))

    val simFactory = MemoryAdapter.factory(Target.Simulation, memoryWords = 512)
    SpinalConfig().generateVerilog(simFactory(axiConfig))
  }

  test("BramAdapter: Write and Read back simulation") {
    val words = 64
    SimConfig.compile(new BramAdapter(axiConfig, memoryWords = words, imgBase = 0x1000, weightBase = 0x2000)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      // Initialize signals
      dut.io.wrEnable #= false
      dut.io.wrAddr #= 0
      dut.io.wrData #= 0
      dut.io.axi.ar.valid #= false
      dut.io.axi.r.ready #= true
      dut.clockDomain.waitSampling(5)

      // 1. Write two words: one in image region, one in weight region
      dut.io.wrEnable #= true
      dut.io.wrAddr #= 0x1000 // imgBase
      dut.io.wrData #= BigInt(0x1122334455667788L)
      dut.clockDomain.waitSampling()

      dut.io.wrAddr #= 0x2000 // weightBase
      dut.io.wrData #= BigInt("AABBCCDDEEFF0011", 16)
      dut.clockDomain.waitSampling()

      dut.io.wrEnable #= false
      dut.clockDomain.waitSampling()

      // 2. Read back image region
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.payload.addr #= 0x1000
      dut.io.axi.ar.payload.len #= 0
      dut.io.axi.ar.payload.id #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.axi.ar.ready.toBoolean)
      dut.io.axi.ar.valid #= false

      dut.clockDomain.waitSamplingWhere(dut.io.axi.r.valid.toBoolean)
      assert(dut.io.axi.r.payload.data.toBigInt == BigInt(0x1122334455667788L))
      assert(dut.io.axi.r.payload.last.toBoolean)

      dut.clockDomain.waitSampling(2)

      // 3. Read back weight region
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.payload.addr #= 0x2000
      dut.io.axi.ar.payload.len #= 0
      dut.io.axi.ar.payload.id #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.axi.ar.ready.toBoolean)
      dut.io.axi.ar.valid #= false

      dut.clockDomain.waitSamplingWhere(dut.io.axi.r.valid.toBoolean)
      assert(dut.io.axi.r.payload.data.toBigInt == BigInt("AABBCCDDEEFF0011", 16))
      assert(dut.io.axi.r.payload.last.toBoolean)
    }
  }

  test("SramAsicAdapter: Write and Read back simulation") {
    val words = 64
    SimConfig.compile(new SramAsicAdapter(axiConfig, memoryWords = words, imgBase = 0x1000, weightBase = 0x2000, pdk = PdkFamily.Sky130)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.wrEnable #= false
      dut.io.wrAddr #= 0
      dut.io.wrData #= 0
      dut.io.axi.ar.valid #= false
      dut.io.axi.r.ready #= true
      dut.clockDomain.waitSampling(5)

      // Write word
      dut.io.wrEnable #= true
      dut.io.wrAddr #= 0x1000
      dut.io.wrData #= BigInt("CAFEBABE12345678", 16)
      dut.clockDomain.waitSampling()

      dut.io.wrEnable #= false
      dut.clockDomain.waitSampling()

      // Read back
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.payload.addr #= 0x1000
      dut.io.axi.ar.payload.len #= 0
      dut.clockDomain.waitSamplingWhere(dut.io.axi.ar.ready.toBoolean)
      dut.io.axi.ar.valid #= false

      dut.clockDomain.waitSamplingWhere(dut.io.axi.r.valid.toBoolean)
      assert(dut.io.axi.r.payload.data.toBigInt == BigInt("CAFEBABE12345678", 16))
      assert(dut.io.axi.r.payload.last.toBoolean)
    }
  }

  test("DdrAdapter: Host write translation to AXI master write") {
    SimConfig.compile(new DdrAdapter(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.wrEnable #= false
      dut.io.wrAddr #= 0
      dut.io.wrData #= 0
      dut.io.axi.ar.valid #= false
      dut.extIo.ddrMaster.aw.ready #= false
      dut.extIo.ddrMaster.w.ready #= false
      dut.extIo.ddrMaster.r.valid #= false
      dut.extIo.ddrMaster.b.valid #= false
      dut.clockDomain.waitSampling(5)

      // Trigger write from host
      dut.io.wrEnable #= true
      dut.io.wrAddr #= 0x80000000L
      dut.io.wrData #= BigInt("DEADBEEF00112233", 16)
      dut.clockDomain.waitSampling()

      dut.io.wrEnable #= false
      dut.clockDomain.waitSampling()

      // Check external DDR master write signals while pending
      assert(dut.extIo.ddrMaster.aw.valid.toBoolean)
      assert(dut.extIo.ddrMaster.aw.payload.addr.toLong == 0x80000000L)
      assert(dut.extIo.ddrMaster.w.valid.toBoolean)
      assert(dut.extIo.ddrMaster.w.payload.data.toBigInt == BigInt("DEADBEEF00112233", 16))
      assert(dut.extIo.ddrMaster.w.payload.last.toBoolean)

      // Now accept the write transaction
      dut.extIo.ddrMaster.aw.ready #= true
      dut.extIo.ddrMaster.w.ready #= true
      dut.clockDomain.waitSampling(2)
      assert(!dut.extIo.ddrMaster.aw.valid.toBoolean)
      assert(!dut.extIo.ddrMaster.w.valid.toBoolean)
    }
  }




  test("UartSoC: Seamless instantiation with custom SramAsicAdapter") {
    val targetDir = "out/test_soc_sram"
    SpinalConfig(targetDirectory = targetDir).generateVerilog(
      new UartSoC(
        acceleratorFactory = () => new Accelerator(
          dataType = I8(),
          inputShape = Seq(4, 4, 1),
          modelSpec = Seq(spinalML.nn.Flatten()),
          axiConfig = axiConfig
        ),
        memoryAdapterFactory = Some((cfg: Axi4Config) => new SramAsicAdapter(cfg, memoryWords = 512, pdk = PdkFamily.Sky130))
      )
    )
    val content = scala.io.Source.fromFile(s"$targetDir/UartSoC.v").mkString
    assert(content.contains("SramAsicAdapter"), "UartSoC Verilog must contain SramAsicAdapter when factory is injected")
  }
}
