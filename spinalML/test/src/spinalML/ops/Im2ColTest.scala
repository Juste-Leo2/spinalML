package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class Im2ColTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(3, 3), lanes = 1)) // 3x3 image
    val c = master(Tensor(I8(), Seq(4, 4), lanes = 4)) // 4 windows of 2x2
  }
  io.c <> im2col(io.a, kernelSize = 2)
}

class Im2ColTest extends AnyFunSuite {
  test("Test Im2Col sliding window logic on 3x3 image with 2x2 kernel") {
    SimConfig.withWave.compile(Im2ColTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      val inputs = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9)
      val expectedOutputs = Seq(
        Seq(1, 2, 4, 5),
        Seq(2, 3, 5, 6),
        Seq(4, 5, 7, 8),
        Seq(5, 6, 8, 9)
      )
      
      // Thread to feed inputs
      fork {
        var i = 0
        dut.io.a.stream.valid #= true
        while (i < 9) {
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
      while (i < 4) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          val w0 = dut.io.c.stream.payload(0).toInt
          val w1 = dut.io.c.stream.payload(1).toInt
          val w2 = dut.io.c.stream.payload(2).toInt
          val w3 = dut.io.c.stream.payload(3).toInt
          
          assert(w0 == expectedOutputs(i)(0), s"Window $i element 0: expected ${expectedOutputs(i)(0)}, got $w0")
          assert(w1 == expectedOutputs(i)(1), s"Window $i element 1: expected ${expectedOutputs(i)(1)}, got $w1")
          assert(w2 == expectedOutputs(i)(2), s"Window $i element 2: expected ${expectedOutputs(i)(2)}, got $w2")
          assert(w3 == expectedOutputs(i)(3), s"Window $i element 3: expected ${expectedOutputs(i)(3)}, got $w3")
          i += 1
        }
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
