package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.dtypes.I8 
import spinalML.tensors.Tensor

object DMAReader2DTest {
  def main(args: Array[String]): Unit = {
    // 1. Configs
    val axiDataWidth = 64        // 64-bit DDR4 bus (4 16-bit elements per beat)
    val outLanes = 1             // The ML layer wants 1 element per clock
    
    val axiConfig = Axi4Config(
      addressWidth = 32,
      dataWidth = axiDataWidth,
      idWidth = 4
    )
    
    // Wrapper to expose full Axi4 for the simulator
    case class DMAReader2DTestWrapper() extends Component {
      val dataType = SInt(16 bits)
      val io = new Bundle {
        val cmd = slave(Stream(FetchRequest2D(32)))
        val axiMaster = master(Axi4(axiConfig))
        val outStream = master(Tensor(dataType, Seq(4, 4), outLanes))
      }
      val reader = DMAReader2D(dataType, Seq(4, 4), outLanes, axiConfig)
      reader.io.cmd << io.cmd
      io.outStream <> reader.io.outStream
      
      io.axiMaster.ar << reader.io.axiMaster.ar
      reader.io.axiMaster.r << io.axiMaster.r
      
      io.axiMaster.aw.valid := False
      io.axiMaster.aw.payload.assignDontCare()
      io.axiMaster.w.valid := False
      io.axiMaster.w.payload.assignDontCare()
      io.axiMaster.b.ready := False
    }

    SimConfig.withVerilator.withFstWave.workspacePath("sim_build").compile {
      val dut = DMAReader2DTestWrapper()
      dut.setDefinitionName("DMAReader2DTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize inputs
      dut.io.cmd.valid #= false
      dut.io.cmd.baseAddress #= 0
      dut.io.cmd.stride #= 0
      dut.io.cmd.patchWidth #= 0
      dut.io.cmd.patchHeight #= 0
      dut.io.outStream.stream.ready #= true
      
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      
      // Let's write a 4x4 image into DDR memory at address 0x1000
      // 4 pixels per row * 2 bytes = 8 bytes per row.
      // We will write 4 rows.
      val baseAddr = 0x1000
      
      // Row 0: 1, 2, 3, 4
      memorySim.memory.writeBigInt(baseAddr + 0, BigInt("0004000300020001", 16), 8)
      // Row 1: 5, 6, 7, 8
      memorySim.memory.writeBigInt(baseAddr + 8, BigInt("0008000700060005", 16), 8)
      // Row 2: 9, 10, 11, 12
      memorySim.memory.writeBigInt(baseAddr + 16, BigInt("000C000B000A0009", 16), 8)
      // Row 3: 13, 14, 15, 16
      memorySim.memory.writeBigInt(baseAddr + 24, BigInt("0010000F000E000D", 16), 8)
      
      memorySim.start()
      
      dut.clockDomain.waitSampling(5)
      
      // Trigger the 2D DMA fetch command!
      dut.io.cmd.valid #= true
      dut.io.cmd.baseAddress #= baseAddr
      dut.io.cmd.stride #= 8 // 8 bytes between each row
      dut.io.cmd.patchWidth #= 0 // N-1 beats (so 1 beat = 4 pixels)
      dut.io.cmd.patchHeight #= 4 // 4 rows
      
      // Fork a thread to drop valid when ready is asserted
      fork {
        dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
        dut.io.cmd.valid #= false
      }
      
      var validCount = 0
      var timeout = 0
      while (validCount < 16 && timeout < 500) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val p0 = dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toBigInt
          println(f"Received Pixel: $p0%d")
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }
      
      assert(validCount == 16, s"Expected 16 valid outputs, got $validCount")
      println("DMAReader2D hardware simulation successful!")
    }
  }
}
