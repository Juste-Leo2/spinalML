// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.test

import org.scalatest.funsuite.AnyFunSuite
import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.AxiMemorySim
import spinalML.examples.SequentialCNN

object SequentialCNNTest {
  def run(): Unit = {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  
  SimConfig.withWave.compile(SequentialCNN(axiConfig)).doSim { dut =>
    dut.clockDomain.forkStimulus(10)
    
    // Setup AXI Memory Sim (our virtual DDR)
    val memorySim = AxiMemorySim(
      axi = dut.io.axiMaster,
      clockDomain = dut.clockDomain,
      config = spinal.lib.bus.amba4.axi.sim.AxiMemorySimConfig(maxOutstandingReads = 8)
    )
    memorySim.start()
    
    // Fill memory with dummy data (e.g., 1s everywhere)
    val imgBase = 0x1000
    val weightBase = 0x2000
    
    // Image: 8x8 = 64 elements. AXI is 64-bit, type is 16-bit -> 4 elements per beat
    for (i <- 0 until (64 / 4)) {
      memorySim.memory.writeBigInt(imgBase + i * 8, BigInt("0001000100010001", 16), 8)
    }
    
    // Weights for Conv2D (9), Bias (1), Linear W (36), Bias (1)
    // Total elements = 9 + 1 + 36 + 1 = 47 elements
    // 47 / 4 = 12 beats. We'll just fill a small chunk to be safe.
    for (i <- 0 until 50) {
      memorySim.memory.writeBigInt(weightBase + i * 8, BigInt("0001000100010001", 16), 8)
    }
    
    // Helper function to write to AXI4-Lite
    def writeAxiLite(addr: BigInt, data: BigInt) = {
      dut.io.ctrlBus.aw.valid #= true
      dut.io.ctrlBus.aw.payload.addr #= addr
      dut.io.ctrlBus.w.valid #= true
      dut.io.ctrlBus.w.payload.data #= data
      dut.io.ctrlBus.w.payload.strb #= 0xF
      dut.io.ctrlBus.b.ready #= true
      
      dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
      dut.io.ctrlBus.b.ready #= false
      dut.clockDomain.waitSampling()
    }
    
    // Init control bus
    dut.io.ctrlBus.aw.valid #= false
    dut.io.ctrlBus.w.valid #= false
    dut.io.ctrlBus.ar.valid #= false
    dut.io.ctrlBus.r.ready #= false
    dut.io.ctrlBus.b.ready #= false
    dut.io.outStream.stream.ready #= true
    
    dut.clockDomain.waitSampling(5)
    
    // Configure Addresses via AXI-Lite
    writeAxiLite(0x08, imgBase)    // Image Base Address
    writeAxiLite(0x0C, weightBase) // Weights Base Address
    
    // Fire the start signal via AXI-Lite
    writeAxiLite(0x00, 1)
    
    var validCount = 0
    var timeout = 0
    // Wait for the single output from the Linear layer
    while (validCount < 1 && timeout < 20000) {
      if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
        validCount += 1
        val outVal = dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toBigInt
        println(f"SequentialCNN Output: $outVal%d")
      }
      dut.clockDomain.waitSampling()
      timeout += 1
    }
    
    assert(validCount == 1, s"Expected 1 output, got $validCount")
    println("SequentialCNN Simulation Successful!")
  }
  }

  def main(args: Array[String]): Unit = run()
}

class SequentialCNNTest extends AnyFunSuite {
  test("SequentialCNN SoC simulation (AXI4 DMA + AXI-Lite control)") {
    SequentialCNNTest.run()
  }
}
