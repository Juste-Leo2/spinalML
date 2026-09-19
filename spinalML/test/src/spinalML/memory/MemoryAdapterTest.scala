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
      dut.io.wrStrb #= 0xFF
      dut.io.axi.ar.valid #= false
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.io.axi.b.ready #= true
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
      dut.io.wrStrb #= 0xFF
      dut.io.axi.ar.valid #= false
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.io.axi.b.ready #= true
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
      dut.io.wrStrb #= 0xFF
      dut.io.axi.ar.valid #= false
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.io.axi.b.ready #= true
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

  test("DdrAdapter: Flow control wrReady backpressures host writes until AXI b.valid") {
    SimConfig.compile(new DdrAdapter(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.wrEnable #= false
      dut.io.wrAddr #= 0
      dut.io.wrData #= 0
      dut.io.wrStrb #= 0xFF
      dut.io.axi.ar.valid #= false
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.io.axi.b.ready #= true
      dut.extIo.ddrMaster.aw.ready #= false
      dut.extIo.ddrMaster.w.ready #= false
      dut.extIo.ddrMaster.r.valid #= false
      dut.extIo.ddrMaster.b.valid #= false
      dut.clockDomain.waitSampling(5)

      // Initially, DDR adapter must be ready for host writes
      assert(dut.io.wrReady.toBoolean, "DdrAdapter must be wrReady initially when idle")

      // First host write
      dut.io.wrEnable #= true
      dut.io.wrAddr #= 0x80000000L
      dut.io.wrData #= BigInt("1111111111111111", 16)
      dut.clockDomain.waitSampling()
      dut.io.wrEnable #= false
      dut.clockDomain.waitSampling()

      assert(!dut.io.wrReady.toBoolean, "DdrAdapter must drop wrReady while host write is in-flight")

      // Attempting an untimely write while !wrReady must NOT clobber in-flight transaction
      dut.io.wrEnable #= true
      dut.io.wrAddr #= 0x80000008L
      dut.io.wrData #= BigInt("2222222222222222", 16)
      dut.clockDomain.waitSampling()
      dut.io.wrEnable #= false

      // In-flight AXI signals must still carry the FIRST write's address and data
      assert(dut.extIo.ddrMaster.aw.valid.toBoolean)
      assert(dut.extIo.ddrMaster.aw.payload.addr.toLong == 0x80000000L, "aw.addr was clobbered by untimely write!")
      assert(dut.extIo.ddrMaster.w.payload.data.toBigInt == BigInt("1111111111111111", 16), "w.data was clobbered by untimely write!")

      // Complete the first transaction on AXI
      dut.extIo.ddrMaster.aw.ready #= true
      dut.extIo.ddrMaster.w.ready #= true
      dut.clockDomain.waitSampling()
      dut.extIo.ddrMaster.aw.ready #= false
      dut.extIo.ddrMaster.w.ready #= false

      // Still waiting for B response: wrReady must remain false
      assert(!dut.io.wrReady.toBoolean, "wrReady must remain false until B response arrives")

      // Send B response
      dut.extIo.ddrMaster.b.valid #= true
      dut.extIo.ddrMaster.b.payload.resp #= 0
      dut.clockDomain.waitSampling()
      dut.extIo.ddrMaster.b.valid #= false
      dut.clockDomain.waitSampling()

      // Now the adapter must be wrReady again!
      assert(dut.io.wrReady.toBoolean, "DdrAdapter must return to wrReady after B response")
    }
  }

  test("BUG-DDR-04: DdrAdapter RAW hazard interlock stalls AXI AR until write completes") {
    SimConfig.compile(new DdrAdapter(axiConfig)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.wrEnable #= false
      dut.io.wrAddr #= 0
      dut.io.wrData #= 0
      dut.io.wrStrb #= 0xFF
      dut.io.axi.ar.valid #= false
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.io.axi.b.ready #= true
      dut.io.axi.r.ready #= true
      dut.extIo.ddrMaster.aw.ready #= false
      dut.extIo.ddrMaster.w.ready #= false
      dut.extIo.ddrMaster.r.valid #= false
      dut.extIo.ddrMaster.b.valid #= false
      dut.extIo.ddrMaster.ar.ready #= true
      dut.clockDomain.waitSampling(5)

      // 1. Accelerator initiates a burst write (e.g. accumulator spill in Wave 6 P1)
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.payload.addr #= 0x4000
      dut.io.axi.aw.payload.len #= 1 // 2 beats
      dut.clockDomain.waitSampling()
      dut.io.axi.aw.valid #= false
      dut.clockDomain.waitSampling()

      // 2. While write is in-flight (accAwPending), an AXI read request arrives (e.g. next pass)
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.payload.addr #= 0x4000
      dut.clockDomain.waitSampling()

      // RAW hazard check: read must NOT be forwarded to DDR master while write is in flight
      assert(!dut.extIo.ddrMaster.ar.valid.toBoolean,
        "RAW Hazard: extIo.ddrMaster.ar.valid must be held low while write is in-flight!")
      assert(!dut.io.axi.ar.ready.toBoolean,
        "RAW Hazard: io.axi.ar.ready must backpressure read master while write is in-flight!")

      // 3. Complete AW handshake on DDR master -> transitions to accStreaming
      dut.extIo.ddrMaster.aw.ready #= true
      dut.clockDomain.waitSampling()
      dut.extIo.ddrMaster.aw.ready #= false
      dut.clockDomain.waitSampling()

      // Read must still be blocked during write streaming
      assert(!dut.extIo.ddrMaster.ar.valid.toBoolean, "ar.valid must remain low during write streaming")
      assert(!dut.io.axi.ar.ready.toBoolean, "ar.ready must remain low during write streaming")

      // 4. Stream write beats
      dut.io.axi.w.valid #= true
      dut.io.axi.w.payload.data #= BigInt("1111222233334444", 16)
      dut.extIo.ddrMaster.w.ready #= true
      dut.clockDomain.waitSampling() // beat 0
      dut.io.axi.w.payload.data #= BigInt("5555666677778888", 16)
      dut.clockDomain.waitSampling() // beat 1 (last)
      dut.io.axi.w.valid #= false
      dut.extIo.ddrMaster.w.ready #= false
      dut.clockDomain.waitSampling()

      // 5. Now in accWaitB state: waiting for DDR write response
      assert(!dut.extIo.ddrMaster.ar.valid.toBoolean, "ar.valid must remain low while waiting for B response")
      assert(!dut.io.axi.ar.ready.toBoolean, "ar.ready must remain low while waiting for B response")

      // 6. DDR controller returns B response
      dut.extIo.ddrMaster.b.valid #= true
      dut.extIo.ddrMaster.b.payload.resp #= 0
      dut.clockDomain.waitSampling()
      dut.extIo.ddrMaster.b.valid #= false
      dut.clockDomain.waitSampling()

      // 7. Write is completed: RAW barrier releases the read request!
      assert(dut.extIo.ddrMaster.ar.valid.toBoolean, "ar.valid must be released once write is complete")
      assert(dut.io.axi.ar.ready.toBoolean, "ar.ready must be released once write is complete")
      dut.io.axi.ar.valid #= false
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

  test("Generate Verilog for Python co-simulation") {
    SpinalConfig().generateVerilog {
      val dut = new BramAdapter(axiConfig, memoryWords = 64, imgBase = 0x1000, weightBase = 0x2000)
      dut.setDefinitionName("BramAdapterTestComp")
      dut
    }
    SpinalConfig().generateVerilog {
      val dut = new SramAsicAdapter(axiConfig, memoryWords = 64, imgBase = 0x1000, weightBase = 0x2000, pdk = PdkFamily.Sky130)
      dut.setDefinitionName("SramAsicAdapterTestComp")
      dut
    }
    SpinalConfig().generateVerilog {
      val dut = new DdrAdapter(axiConfig)
      dut.setDefinitionName("DdrAdapterTestComp")
      dut
    }
  }
}
