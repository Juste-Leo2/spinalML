package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.dtypes.I8 // or we can just use SInt
import spinalML.tensors.Tensor

object DMAReaderTest {
  def main(args: Array[String]): Unit = {
    // 1. Configs
    val axiDataWidth = 64        // 64-bit DDR4 bus (4 16-bit elements per beat)
    val outLanes = 2             // The ML layer only wants 2 elements per clock
    
    val axiConfig = Axi4Config(
      addressWidth = 32,
      dataWidth = axiDataWidth,
      idWidth = 4
    )
    
    // Wrapper to expose full Axi4 for the simulator
    case class DMAReaderTestWrapper() extends Component {
      val dataType = SInt(16 bits)
      val io = new Bundle {
        val cmd = slave(Stream(FetchRequest(32)))
        val axiMaster = master(Axi4(axiConfig))
        val outStream = master(Tensor(dataType, Seq(16), outLanes))
      }
      val reader = DMAReader(dataType, Seq(16), outLanes, axiConfig)
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
      val dut = DMAReaderTestWrapper()
      dut.setDefinitionName("DMAReaderTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize inputs
      dut.io.cmd.valid #= false
      dut.io.cmd.address #= 0
      dut.io.cmd.length #= 0
      dut.io.outStream.stream.ready #= true
      
      // We create a full AXI4 bus for the memory simulator
      // and manually wire the AR/R channels to our DUT's Axi4ReadOnly
      // (This is a software-side simulated wire)
      
      // SpinalHDL's AxiMemorySim works on a simulated physical memory array
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      
      // Let's write some data into our simulated DDR memory at address 0x1000
      // 64-bit per word, so 8 bytes per write.
      // We will write 4 beats = 32 bytes = 16 BF16 elements.
      val baseAddr = 0x1000
      
      memorySim.memory.writeBigInt(baseAddr + 0, BigInt("1111222233334444", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 8, BigInt("5555666677778888", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 16, BigInt("9999AAAABBBBCCCC", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 24, BigInt("DDDDEEEEFFFF0000", 16), 8)
      
      memorySim.start()
      
      dut.clockDomain.waitSampling(5)
      
      // Trigger the DMA fetch command!
      // We want 4 beats of AXI (length = 3, since length is N-1)
      dut.io.cmd.valid #= true
      dut.io.cmd.address #= baseAddr
      dut.io.cmd.length #= 3
      
      dut.clockDomain.waitSampling()
      dut.io.cmd.valid #= false
      
      // Wait and observe the `outStream`
      // Since `outLanes` = 2, and we have 16 BF16 elements, it should take 8 valid cycles to output everything.
      var validCount = 0
      for (_ <- 0 until 50) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val p0 = dut.io.outStream.stream.payload(0).toBigInt
          val p1 = dut.io.outStream.stream.payload(1).toBigInt
          println(f"Received ML Payload: $p1%04X, $p0%04X")
        }
        dut.clockDomain.waitSampling()
      }
      
      assert(validCount == 8, s"Expected 8 valid outputs, got $validCount")
      println("DMAReader hardware simulation successful!")
    }
  }
}
