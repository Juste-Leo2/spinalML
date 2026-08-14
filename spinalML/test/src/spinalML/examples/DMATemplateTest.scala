package spinalML.examples

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.dtypes.I8 // Using SInt/I8 for easy simulation read

object DMATemplateTest {
  def main(args: Array[String]): Unit = {
    // 1. Configs
    val axiDataWidth = 64
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = axiDataWidth, idWidth = 4)
    
    // Wrapper to expose full Axi4 for the simulator
    case class DMATemplateTestWrapper() extends Component {
      val dataType = SInt(16 bits)
      val io = new Bundle {
        val start = slave(Event)
        val imgAddr = in UInt(32 bits)
        val weightAddr = in UInt(32 bits)
        val biasAddr = in UInt(32 bits)
        val axiMaster = master(Axi4(axiConfig))
        val outStream = master(cloneOf(DMATemplate(dataType).io.outStream))
      }
      val template = DMATemplate(dataType)
      template.io.start << io.start
      template.io.imgAddr := io.imgAddr
      template.io.weightAddr := io.weightAddr
      template.io.biasAddr := io.biasAddr
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
      val dut = DMATemplateTestWrapper()
      dut.setDefinitionName("DMATemplateTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize inputs
      dut.io.start.valid #= false
      dut.io.imgAddr #= 0x1000
      dut.io.weightAddr #= 0x2000
      dut.io.biasAddr #= 0x3000
      dut.io.outStream.stream.ready #= true
      
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      
      // Write data to memory
      // We are writing 16-bit values, but AxiMemorySim writes in bytes (width=8)
      // It's easier to write 64-bit BigInts (4 values of 16-bit at a time)
      
      // Image: 8x8 = 64 elements = 16 beats of 64 bits
      val imgBase = 0x1000
      for (i <- 0 until 16) {
        // We write 0x0001000100010001 everywhere (all 1s)
        memorySim.memory.writeBigInt(imgBase + i * 8, BigInt("0001000100010001", 16), 8)
      }
      
      // Weights: 3x3 = 9 elements = 3 beats
      val weightBase = 0x2000
      // We write 0x0001000100010001 (all 1s)
      for (i <- 0 until 3) {
        memorySim.memory.writeBigInt(weightBase + i * 8, BigInt("0001000100010001", 16), 8)
      }
      
      // Bias: 1 element = 1 beat
      val biasBase = 0x3000
      memorySim.memory.writeBigInt(biasBase, BigInt("0000000000000000", 16), 8) // Bias = 0
      
      memorySim.start()
      
      dut.clockDomain.waitSampling(5)
      
      // Trigger the DMA fetch command! We wait for ready to be asserted.
      dut.io.start.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.start.ready.toBoolean)
      dut.io.start.valid #= false
      
      var validCount = 0
      var timeout = 0
      while (validCount < 36 && timeout < 5000) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val outVal = dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toBigInt
          println(f"Conv2D Output: $outVal%d")
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }
      
      assert(validCount == 36, s"Expected 36 outputs, got $validCount")
      println("DMATemplate hardware simulation successful!")
    }
  }
}
