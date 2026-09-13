// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.tensors.Tensor
import org.scalatest.funsuite.AnyFunSuite

// Test wrapper exposing full Axi4 for AxiMemorySim
case class DMAWriterTestWrapper(maxBurstBeats: Int = 4) extends Component {
  val axiDataWidth = 64
  val inLanes = 2
  val axiConfig = Axi4Config(
    addressWidth = 32,
    dataWidth = axiDataWidth,
    idWidth = 4
  )

  val dataType = SInt(16 bits)
  val io = new Bundle {
    val cmd       = slave(Stream(WriteRequest(32)))
    val inStream  = slave(Tensor(dataType, Seq(16), inLanes))
    val axiMaster = master(Axi4(axiConfig))
    val busy      = out Bool()
    val done      = out Bool()
  }

  val writer = DMAWriter(dataType, Seq(16), inLanes, axiConfig, maxBurstBeats = maxBurstBeats)
  writer.io.cmd << io.cmd
  writer.io.inStream <> io.inStream
  io.busy := writer.io.busy
  io.done := writer.io.done

  io.axiMaster.aw << writer.io.axiMaster.aw
  io.axiMaster.w  << writer.io.axiMaster.w
  writer.io.axiMaster.b << io.axiMaster.b

  // Read channels tied off
  io.axiMaster.ar.valid := False
  io.axiMaster.ar.payload.assignDontCare()
  io.axiMaster.r.ready := False
}

class DMAWriterTest extends AnyFunSuite {

  test("DMAWriter Hardware Sim - Basic Burst Write") {
    SimConfig.withVerilator.workspacePath("sim_build").compile {
      val dut = DMAWriterTestWrapper(maxBurstBeats = 16)
      dut.setDefinitionName("DMAWriterTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.cmd.valid #= false
      dut.io.cmd.address #= 0
      dut.io.cmd.length #= 0
      dut.io.inStream.stream.valid #= false

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      memorySim.start()
      dut.clockDomain.waitSampling(5)

      // Test data: 16 elements of 16-bit signed integers [-32768, 32767]
      // 16 elements = 32 bytes = 4 beats on 64-bit bus
      val testValues = (0 until 16).map(i => BigInt((i + 1) * 1000))
      val baseAddr = 0x1000

      // Send command: 4 beats (length = 3)
      dut.io.cmd.valid #= true
      dut.io.cmd.address #= baseAddr
      dut.io.cmd.length #= 3
      dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
      dut.io.cmd.valid #= false

      // Stream the 16 elements in pairs of 2 (inLanes = 2, 8 stream cycles)
      var elemIdx = 0
      while (elemIdx < 16) {
        dut.io.inStream.stream.valid #= true
        dut.io.inStream.stream.payload(0) #= testValues(elemIdx)
        dut.io.inStream.stream.payload(1) #= testValues(elemIdx + 1)
        dut.clockDomain.waitSamplingWhere(dut.io.inStream.stream.ready.toBoolean)
        elemIdx += 2
      }
      dut.io.inStream.stream.valid #= false

      // Wait for completion (done pulse or busy falling)
      var cycles = 0
      while (dut.io.busy.toBoolean && cycles < 100) {
        dut.clockDomain.waitSampling()
        cycles += 1
      }
      assert(!dut.io.busy.toBoolean, "DMAWriter timed out while busy")

      // Verify DDR memory contents
      // 16 elements of 16 bits = 4 words of 64 bits
      for (w <- 0 until 4) {
        val wordAddr = baseAddr + w * 8
        val memWord = memorySim.memory.readBigInt(wordAddr, 8)
        val expectedWord = (0 until 4).map { b =>
          val elem = testValues(w * 4 + b) & 0xFFFF
          elem << (b * 16)
        }.reduce(_ | _)

        assert(memWord == expectedWord,
          f"Word $w at 0x$wordAddr%X: read 0x$memWord%016X != expected 0x$expectedWord%016X")
      }
    }
  }

  test("DMAWriter Hardware Sim - 4KB Boundary Split") {
    // Start at 0x1FF0 (16 bytes before 4KB boundary 0x2000)
    // Writing 32 bytes (4 beats) must be split into:
    //  - Burst 1: 2 beats (0x1FF0 to 0x2000)
    //  - Burst 2: 2 beats (0x2000 to 0x2010)
    SimConfig.withVerilator.workspacePath("sim_build").compile {
      val dut = DMAWriterTestWrapper(maxBurstBeats = 16)
      dut.setDefinitionName("DMAWriterBoundaryTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.cmd.valid #= false
      dut.io.cmd.address #= 0
      dut.io.cmd.length #= 0
      dut.io.inStream.stream.valid #= false

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      memorySim.start()
      dut.clockDomain.waitSampling(5)

      val testValues = (0 until 16).map(i => BigInt((i + 0x10) * 0x0101))
      val boundaryAddr = 0x1FF0

      dut.io.cmd.valid #= true
      dut.io.cmd.address #= boundaryAddr
      dut.io.cmd.length #= 3
      dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
      dut.io.cmd.valid #= false

      var elemIdx = 0
      while (elemIdx < 16) {
        dut.io.inStream.stream.valid #= true
        dut.io.inStream.stream.payload(0) #= testValues(elemIdx)
        dut.io.inStream.stream.payload(1) #= testValues(elemIdx + 1)
        dut.clockDomain.waitSamplingWhere(dut.io.inStream.stream.ready.toBoolean)
        elemIdx += 2
      }
      dut.io.inStream.stream.valid #= false

      var cycles = 0
      while (dut.io.busy.toBoolean && cycles < 100) {
        dut.clockDomain.waitSampling()
        cycles += 1
      }
      assert(!dut.io.busy.toBoolean, "DMAWriter timed out on boundary test")

      for (w <- 0 until 4) {
        val wordAddr = boundaryAddr + w * 8
        val memWord = memorySim.memory.readBigInt(wordAddr, 8)
        val expectedWord = (0 until 4).map { b =>
          val elem = testValues(w * 4 + b) & 0xFFFF
          elem << (b * 16)
        }.reduce(_ | _)

        assert(memWord == expectedWord,
          f"Boundary Word $w at 0x$wordAddr%X: read 0x$memWord%016X != expected 0x$expectedWord%016X")
      }
    }
  }

  test("Generate Verilog for DMAWriterTestWrapper") {
    SpinalConfig().generateVerilog {
      val dut = DMAWriterTestWrapper()
      dut.setDefinitionName("DMAWriterTestComp")
      dut
    }
  }
}
