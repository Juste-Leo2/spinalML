package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Slice Axis 0
case class SliceTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(4), lanes = 2))
    val c = master(Tensor(I8(), Seq(2), lanes = 2))
  }
  
  // Keep chunks 1 and 2, drop 0 and 3
  io.c <> spinalML.ops.slice(io.a, start = 1, end = 3, axis = 0)
}

class SliceTest extends AnyFunSuite {
  test("Test streaming Slice operation on Axis 0") {
    SimConfig.withWave.compile(SliceTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed A
      fork {
        for(i <- 0 until 4) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= i + 10
          dut.io.a.stream.payload(1) #= i + 20
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        }
        dut.io.a.stream.valid #= false
      }
      
      // Check results (we expect i=1 and i=2 to pass through)
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 11)
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 12)
      
      dut.clockDomain.waitSampling(10)
    }
  }
}
