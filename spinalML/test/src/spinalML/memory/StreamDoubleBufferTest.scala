package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.I32

class StreamDoubleBufferTest extends AnyFunSuite {
  test("Double Buffering Ping-Pong logic") {
    SimConfig.withWave.compile {
      StreamDoubleBuffer(I32(), depth = 8, lanes = 2) // 4 addresses per bank
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      
      dut.io.streamIn.valid #= false
      dut.io.nextTile #= false
      dut.io.readAddr #= 0
      dut.clockDomain.waitSampling()
      
      // Load Ping bank (4 writes of 2 lanes)
      for (i <- 0 until 4) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= i * 2
        dut.io.streamIn.payload(1) #= i * 2 + 1
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      dut.clockDomain.waitSampling() // Wait for pingFull register to update
      
      assert(dut.io.tileReady.toBoolean == true, "Ping bank should be ready")
      
      // Read from Ping bank
      for (i <- 0 until 4) {
        dut.io.readAddr #= i
        dut.clockDomain.waitSampling() // wait for readAddr to register
        // Since readSync has 1 cycle latency, we need an extra cycle before data is valid
        dut.clockDomain.waitSampling() 
        assert(dut.io.readData(0).toInt == i * 2)
        assert(dut.io.readData(1).toInt == i * 2 + 1)
      }
      
      // While reading Ping, load Pong
      for (i <- 0 until 4) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= 100 + i * 2
        dut.io.streamIn.payload(1) #= 100 + i * 2 + 1
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      
      // Finish Ping, switch to Pong
      dut.io.nextTile #= true
      dut.clockDomain.waitSampling()
      dut.io.nextTile #= false
      
      // Read from Pong
      dut.io.readAddr #= 0
      dut.clockDomain.waitSampling()
      dut.clockDomain.waitSampling() // latency
      assert(dut.io.readData(0).toInt == 100)
    }
  }
}
