package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing Slice Axis 0
case class SliceTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  
  // Keep chunks 1 and 2, drop 0 and 3
  io.c <> spinalML.ops.slice(io.a, start = 1, end = 3, axis = 0)
}

class SliceTest extends AnyFunSuite {
  test("Test streaming Slice operation on Axis 0") {
    SimConfig.withWave.compile(SliceTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Feed A
      fork {
        for(i <- 0 until 4) {
          dut.io.a.stream.valid #= true
          dut.io.a.stream.payload(0) #= i + 1
          dut.io.a.stream.payload(1) #= i + 2
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        }
        dut.io.a.stream.valid #= false
      }
      
      // Check results (we expect i=1 and i=2 to pass through)
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 2)
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 3)
      
      dut.clockDomain.waitSampling(10)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test Slice compilation on $name") {
      SpinalConfig().generateVerilog(SliceTestComp(dt()))
    }
  }
}
