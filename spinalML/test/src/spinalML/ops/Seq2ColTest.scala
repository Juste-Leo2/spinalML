package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class Seq2ColTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I4(), Seq(3, 1), lanes = 1)) // Sequence of 3
    val c = master(Tensor(I4(), Seq(2, 2), lanes = 2)) // 2 windows of size 2
  }
  io.c <> seq2col(io.a, kernelSize = 2)
}

class Seq2ColTest extends AnyFunSuite {
  test("Test Seq2Col sliding window logic") {
    SimConfig.withWave.compile(Seq2ColTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      val inputs = Seq(1, 2, 3)
      val expectedOutputs = Seq(
        Seq(1, 2),
        Seq(2, 3)
      )
      
      // Thread to feed inputs
      fork {
        var i = 0
        dut.io.a.stream.valid #= true
        while (i < 3) {
          dut.io.a.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSampling()
          if (dut.io.a.stream.ready.toBoolean) {
            i += 1
          }
        }
        dut.io.a.stream.valid #= false
      }
      
      // Thread to check outputs
      var i = 0
      while (i < 2) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          val w0 = dut.io.c.stream.payload(0).toInt
          val w1 = dut.io.c.stream.payload(1).toInt
          
          assert(w0 == expectedOutputs(i)(0), s"Window $i element 0: expected ${expectedOutputs(i)(0)}, got $w0")
          assert(w1 == expectedOutputs(i)(1), s"Window $i element 1: expected ${expectedOutputs(i)(1)}, got $w1")
          i += 1
        }
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Seq2Col compilation on FP4") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP4_E2M1(), Seq(3, 1), lanes = 1))
      val c = master(Tensor(FP4_E2M1(), Seq(2, 2), lanes = 2))
      c <> seq2col(a, kernelSize = 2)
    })
  }
}
