package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.dtypes.I8 
import spinalML.tensors.Tensor
import org.scalatest.funsuite.AnyFunSuite

// Wrapper to expose full Axi4 for the simulator and Python Cocotb
case class DMAReaderTestWrapper() extends Component {
  val axiDataWidth = 64
  val outLanes = 2
  val axiConfig = Axi4Config(
    addressWidth = 32,
    dataWidth = axiDataWidth,
    idWidth = 4
  )
  
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

class DMAReaderTest extends AnyFunSuite {

  test("DMAReader Hardware Sim") {
    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile {
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
      
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      
      val baseAddr = 0x1000
      memorySim.memory.writeBigInt(baseAddr + 0, BigInt("1111222233334444", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 8, BigInt("5555666677778888", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 16, BigInt("9999AAAABBBBCCCC", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 24, BigInt("DDDDEEEEFFFF0000", 16), 8)
      
      memorySim.start()
      dut.clockDomain.waitSampling(5)
      
      dut.io.cmd.valid #= true
      dut.io.cmd.address #= baseAddr
      dut.io.cmd.length #= 3
      
      dut.clockDomain.waitSampling()
      dut.io.cmd.valid #= false
      
      var validCount = 0
      for (_ <- 0 until 50) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val p0 = dut.io.outStream.stream.payload(0).toBigInt
          val p1 = dut.io.outStream.stream.payload(1).toBigInt
        }
        dut.clockDomain.waitSampling()
      }
      
      assert(validCount == 8, s"Expected 8 valid outputs, got $validCount")
    }
  }

  test("Generate Verilog for Python Cocotb") {
    SpinalConfig().generateVerilog {
      val dut = DMAReaderTestWrapper()
      dut.setDefinitionName("DMAReaderTestComp")
      dut
    }
  }
}
