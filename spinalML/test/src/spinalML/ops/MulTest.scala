package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the mul operation
case class MulTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(4), lanes = 2))
    val b = slave(Tensor(I8(), Seq(4), lanes = 2))
    val c = master(Tensor(I8(), Seq(4), lanes = 2))
  }
  
  // GGML-like syntax for mul
  io.c <> spinalML.ops.mul(io.a, io.b)
}

class MulTest extends AnyFunSuite {
  test("Test streaming pipelined mul operation on I8 tensors") {
    SimConfig.withWave.compile(MulTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk (lanes = 2)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 10
      dut.io.a.stream.payload(1) #= -5
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 2
      dut.io.b.stream.payload(1) #= 4
      
      // Because of m2sPipe, the result will take 1 cycle to arrive.
      // waitSamplingWhere will correctly wait until the output is valid.
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Check results for chunk 1 (10*2 = 20, -5*4 = -20)
      assert(dut.io.c.stream.payload(0).toInt == 20)
      assert(dut.io.c.stream.payload(1).toInt == -20)
      
      // Stop input stream to ensure pipeline flushes properly
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.clockDomain.waitSampling(3)
    }
  }
}
