package spinalML.examples

import spinal.core._
import spinalML.dtypes.FloatML

object SimpleCNNTest {
  def main(args: Array[String]): Unit = {
    // SimpleCNN uses BF16 by default for this example test
    import spinal.core.sim._

    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile({
      SimpleCNN(FloatML(8, 7)).setDefinitionName("SimpleCNNTestComp")
    }).doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling(5)
      println("SimpleCNN simulation successful!")
    }
  }
}
