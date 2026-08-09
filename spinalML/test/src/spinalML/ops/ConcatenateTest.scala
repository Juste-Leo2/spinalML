package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Concatenate Axis 0
case class ConcatenateTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I4(), Seq(2), lanes = 2))
    val b = slave(Tensor(I4(), Seq(4), lanes = 2))
    val c = master(Tensor(I4(), Seq(6), lanes = 2))
  }
  
  io.c <> spinalML.ops.concatenate(io.a, io.b, axis = 0)
}

class ConcatenateTest extends AnyFunSuite {
  test("Test streaming Concatenate operation on Axis 0") {
    SimConfig.withWave.compile(ConcatenateTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed A
      fork {
        for(i <- 0 until 2) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= i + 1
          dut.io.a.stream.payload(1) #= i + 2
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        }
        dut.io.a.stream.valid #= false
      }
      
      // Feed B (concurrently, should be blocked until A is done)
      fork {
        for(i <- 0 until 4) {
          dut.io.b.stream.valid #= true
          dut.io.b.stream.payload(0) #= i + 3
          dut.io.b.stream.payload(1) #= i + 4
          dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
        }
        dut.io.b.stream.valid #= false
      }
      
      // Check results
      for(i <- 0 until 2) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
        assert(dut.io.c.stream.payload(0).toInt == i + 1)
      }
      for(i <- 0 until 4) {
        dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
        assert(dut.io.c.stream.payload(0).toInt == i + 3)
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Concatenate compilation on FP4") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP4_E2M1(), Seq(2), lanes = 2))
      val b = slave(Tensor(FP4_E2M1(), Seq(4), lanes = 2))
      val c = master(Tensor(FP4_E2M1(), Seq(6), lanes = 2))
      c <> spinalML.ops.concatenate(a, b, axis = 0)
    })
  }
}
