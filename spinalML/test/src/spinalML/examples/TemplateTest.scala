// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import spinal.core.sim._
import spinalML.dtypes.FloatML

/**
 * A minimalist testbench for the Template component.
 */
object TemplateTest {
  def main(args: Array[String]): Unit = {
    
    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile({
      // BF16 test datatype
      val dataType = FloatML(expBits = 8, mantBits = 7)
      Template(dataType, shape = Seq(16), lanes = 4)
    }).doSim { dut =>
      
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling(10)
      
      println("Template hardware simulation successful!")
    }
  }
}
