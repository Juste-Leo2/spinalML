package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing repack: converting lanes=2 to lanes=4
case class RepackTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4), lanes = 2))
    val c = master(Tensor(dataType, Seq(4), lanes = 4))
  }
  
  // Use repack to reshape from 2 lanes to 4 lanes
  io.c <> spinalML.ops.repack(io.a, newLanes = 4)
}

class RepackTest extends AnyFunSuite {
  test("Test streaming repack operation (Gearbox) from 2 lanes to 4 lanes") {
    SimConfig.withWave.compile(RepackTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize Stream signals
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Send first chunk of 2 elements (cycle 1)
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 10
      dut.io.a.stream.payload(1) #= 20
      
      // Output should not be valid yet because it needs 4 elements
      assert(dut.io.c.stream.valid.toBoolean == false)
      
      dut.clockDomain.waitSampling()
      
      // Send second chunk of 2 elements (cycle 2)
      dut.io.a.stream.payload(0) #= 30
      dut.io.a.stream.payload(1) #= 40
      
      // Now the output should accumulate and become valid with 4 elements
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      // Check results: it should output [10, 20, 30, 40] in one cycle
      assert(dut.io.c.stream.payload(0).toInt == 10)
      assert(dut.io.c.stream.payload(1).toInt == 20)
      assert(dut.io.c.stream.payload(2).toInt == 30)
      assert(dut.io.c.stream.payload(3).toInt == 40)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test Repack compilation on $name") {
      SpinalConfig().generateVerilog(RepackTestComp(dt()))
    }
  }
}
