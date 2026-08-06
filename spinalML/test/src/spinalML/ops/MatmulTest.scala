package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import org.scalatest.funsuite.AnyFunSuite

// Component for testing matmul: Matrix A [2, 4] x Vector B [4, 1]
case class MatmulTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(2, 4), lanes = 2))
    val b = slave(Tensor(I8(), Seq(4, 1), lanes = 2))
    val c = master(Tensor(I8(), Seq(2, 1), lanes = 1))
  }
  
  io.c <> spinalML.ops.matmul(io.a, io.b)
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
      // B = [1, 2, 3, 4]T
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 1
      dut.io.b.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      
      dut.io.b.stream.valid #= false
      
      // Step 2: Stream Matrix A to compute
      // Row 0: [1, 2, 3, 4]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.payload(0) #= 3
      dut.io.a.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Wait for output C for Row 0
      // 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 30)
      
      // Consume output
      dut.clockDomain.waitSampling()
      
      // Row 1: [-1, -2, -3, -4]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= -1
      dut.io.a.stream.payload(1) #= -2
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.payload(0) #= -3
      dut.io.a.stream.payload(1) #= -4
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Wait for output C for Row 1
      // -1*1 - 2*2 - 3*3 - 4*4 = -1 - 4 - 9 - 16 = -30
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == -30)
      
      dut.clockDomain.waitSampling(5)
    }
  }
}
