package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing matmul: Matrix A [1, 2] x Vector B [2, 1]
case class MatmulTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(1, 2), lanes = 2))
    val b = slave(Tensor(I8(), Seq(2, 1), lanes = 2))
    val c = master(Tensor(I8(), Seq(1, 1), lanes = 1))
  }
  
  // tileSize = 2
  io.c <> spinalML.ops.matmul(io.a, io.b, tileSize = 2)
}

class MatmulTest extends AnyFunSuite {
  test("Test streaming matmul (SRAM + MAC) operation on I8 tensors") {
    SimConfig.withWave.compile(MatmulTestComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load matrix B into internal SRAM
      // B = [3, 4]T
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.valid #= false
      
      // Step 2: Stream Matrix A to compute
      // Row 0: [1, 2]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 3: Wait for output C
      // 1*3 + 2*4 = 3 + 8 = 11
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 11)
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
