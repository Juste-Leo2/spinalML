package spinalML

import spinal.core.sim._
import org.scalatest.funsuite.AnyFunSuite

class HelloWorldSim extends AnyFunSuite {
  test("HelloWorld should add two numbers correctly") {
    SimConfig.withWave.withVerilator.compile(HelloWorld()).doSim { dut =>
      dut.io.a #= 42
      dut.io.b #= 10

      // Advance simulation time
      sleep(1)

      // Check result
      assert(dut.io.result.toInt == 52, s"Expected 52, got ${dut.io.result.toInt}")
      
      println("Simulation passed: 42 + 10 = 52")
    }
  }
}
