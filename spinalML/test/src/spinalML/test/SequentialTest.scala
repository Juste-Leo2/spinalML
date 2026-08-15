package spinalML.test

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.nn._
import spinalML.tensors.Tensor

object SequentialTest {
  def main(args: Array[String]): Unit = {
    // 1. Configs
    val axiDataWidth = 64
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = axiDataWidth, idWidth = 4)
    
    // Wrapper to expose full Axi4 for the simulator
    case class SequentialTestWrapper() extends Component {
      val dataType = HardType(SInt(16 bits)).asInstanceOf[HardType[Data]]
      val inputShape = Seq(8, 8, 1) // H=8, W=8, C=1
      val layers = Seq(
        Conv2D(inChannels=1, outChannels=1, kernelSize=3),
        ReLU(),
        Linear(inFeatures=36, outFeatures=1) // 8x8 -> 6x6 = 36 features
      )
      
      val io = new Bundle {
        val start = slave(Event)
        val imgAddr = in UInt(32 bits)
        val weightAddr = in UInt(32 bits)
        val axiMaster = master(Axi4(axiConfig))
        
        // Output from Linear is 2 elements
        val outStream = master(Tensor(dataType, Seq(1, 1), lanes = 1))
      }
      
      val template = Sequential(dataType, inputShape, layers, axiConfig)
      
      template.io.start << io.start
      template.io.imgBaseAddress := io.imgAddr
      template.io.weightsBaseAddress := io.weightAddr
      io.outStream <> template.io.outStream
      
      io.axiMaster.ar << template.io.axiMaster.ar
      template.io.axiMaster.r << io.axiMaster.r
      
      io.axiMaster.aw.valid := False
      io.axiMaster.aw.payload.assignDontCare()
      io.axiMaster.w.valid := False
      io.axiMaster.w.payload.assignDontCare()
      io.axiMaster.b.ready := False
    }

    SimConfig.withVerilator.withFstWave.workspacePath("sim_build").compile {
      val dut = SequentialTestWrapper()
      dut.setDefinitionName("SequentialTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize inputs
      dut.io.start.valid #= false
      dut.io.imgAddr #= 0x1000
      dut.io.weightAddr #= 0x2000
      dut.io.outStream.stream.ready #= true
      
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      
      // Write data to memory
      val imgBase = 0x1000
      for (i <- 0 until 16) {
        memorySim.memory.writeBigInt(imgBase + i * 8, BigInt("0001000100010001", 16), 8)
      }
      
      val weightBase = 0x2000
      for (i <- 0 until 100) { 
        memorySim.memory.writeBigInt(weightBase + i * 8, BigInt("0001000100010001", 16), 8)
      }
      
      memorySim.start()
      dut.clockDomain.waitSampling(5)
      
      // Trigger the computation
      dut.io.start.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.start.ready.toBoolean)
      dut.io.start.valid #= false
      
      var validCount = 0
      var timeout = 0
      while (validCount < 1 && timeout < 5000) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val outVal = dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toBigInt
          println(f"Sequential Output: $outVal%d")
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }
      
      assert(validCount == 1, s"Expected 1 outputs, got $validCount")
      println("Sequential hardware simulation successful!")
    }
  }
}
