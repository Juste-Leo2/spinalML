package spinalML.examples

import spinal.core.sim._
import spinalML.dtypes.FloatML

/**
 * A minimalist testbench for the Template component.
 */
object TemplateTest {
  def main(args: Array[String]): Unit = {
    
    // 1. Compile the component into a Verilator simulation model
    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile({
      // We choose BF16 as our test datatype
      val dataType = FloatML(expBits = 8, mantBits = 7)
      Template(dataType, shape = Seq(16), lanes = 4)
    }).doSim { dut =>
      
      // 2. Start the hardware clock
      dut.clockDomain.forkStimulus(period = 10)
      
      // 3. Initialize IOs
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      // 4. Run the simulation for a few cycles
      dut.clockDomain.waitSampling(10)
      
      println("Template hardware simulation successful!")
    }
  }
}
