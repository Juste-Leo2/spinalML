package spinalML.test

import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.AxiMemorySim
import spinalML.examples.Comprehensive1DCNN

object Comprehensive1DCNNTest extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  
  SimConfig.withWave.compile(Comprehensive1DCNN(axiConfig)).doSim { dut =>
    dut.clockDomain.forkStimulus(10)
    
    val memorySim = AxiMemorySim(
      axi = dut.io.axiMaster,
      clockDomain = dut.clockDomain,
      config = spinal.lib.bus.amba4.axi.sim.AxiMemorySimConfig(maxOutstandingReads = 8)
    )
    memorySim.start()
    
    val imgBase = 0x1000
    val weightBase = 0x2000
    
    // Fill memory with dummy data (e.g., 1s everywhere)
    for (i <- 0 until 100) {
      memorySim.memory.writeBigInt(imgBase + i * 8, BigInt("0001000100010001", 16), 8)
      memorySim.memory.writeBigInt(weightBase + i * 8, BigInt("0001000100010001", 16), 8)
    }
    
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
    
    dut.io.ctrlBus.aw.valid #= false
    dut.io.ctrlBus.w.valid #= false
    dut.io.ctrlBus.ar.valid #= false
    dut.io.ctrlBus.r.ready #= false
    dut.io.ctrlBus.b.ready #= false
    dut.io.outStream.stream.ready #= true
    
    dut.clockDomain.waitSampling(5)
    
    writeAxiLite(0x08, imgBase)
    writeAxiLite(0x0C, weightBase)
    writeAxiLite(0x00, 1)
    
    var validCount = 0
    var timeout = 0
    while (validCount < 1 && timeout < 20000) { 
      if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
        validCount += 1
        // Usually lanes = 1 at output of Sequential for now unless modified.
        // We will just read the output and log it.
        println(f"Comprehensive1DCNN Output received!")
      }
      dut.clockDomain.waitSampling()
      timeout += 1
    }
    
    assert(validCount == 1, s"Expected 1 output stream transaction, got $validCount")
    println("Comprehensive1DCNN Simulation Successful!")
  }
}
