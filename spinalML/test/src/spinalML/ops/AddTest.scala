package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing the add operation
case class AddTestComp() extends Component {
  val io = new Bundle {
    val a = in(Tensor(I8(), Seq(2)))
    val b = in(Tensor(I8(), Seq(2)))
    val c = out(Tensor(I8(), Seq(2)))
  }
  
  // GGML-like syntax for add
  io.c := spinalML.ops.add(io.a, io.b)
}

class AddTest extends AnyFunSuite {
  test("Test add operation on I8 tensors") {
    SimConfig.withWave.compile(AddTestComp()).doSim { dut =>
      // Initialize inputs
      dut.io.a.data(0) #= 10
      dut.io.a.data(1) #= -5
      
      dut.io.b.data(0) #= 15
      dut.io.b.data(1) #= 10
      
      sleep(1)
      
      // Check results (10+15 = 25, -5+10 = 5)
      assert(dut.io.c.data(0).toInt == 25)
      assert(dut.io.c.data(1).toInt == 5)
    }
  }
}
