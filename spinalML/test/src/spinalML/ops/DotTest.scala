package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Component for testing dot product: two 8-element vectors streamed on 2 lanes
case class DotTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(8), lanes = 2))
    val b = slave(Tensor(dataType, Seq(8), lanes = 2))
    val c = master(Tensor(dataType, Seq(1), lanes = 1))
  }
  io.c <> spinalML.ops.dot(io.a, io.b)
}

class DotTest extends AnyFunSuite {
  test("Test streaming dot product operation on I8 tensors") {
    SimConfig.withWave.compile(DotTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true

      dut.clockDomain.waitSampling()

      // Step 1: Load vector B into the internal buffer
      // B = [1, 2, 3, 4, 5, 6, 7, 8]
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 1
      dut.io.b.stream.payload(1) #= 2
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.payload(0) #= 3
      dut.io.b.stream.payload(1) #= 4
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.payload(0) #= 5
      dut.io.b.stream.payload(1) #= 6
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.payload(0) #= 7
      dut.io.b.stream.payload(1) #= 8
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false

      // Step 2: Stream vector A to compute
      // A = [1, 1, 1, 1, 1, 1, 1, 1]
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      dut.io.a.stream.payload(0) #= 1
      dut.io.a.stream.payload(1) #= 1
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      dut.io.a.stream.valid #= false

      // Step 3: Wait for the scalar output C
      // 1*1 + 2*1 + ... + 8*1 = 36
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean)
      assert(dut.io.c.stream.payload(0).toInt == 36)

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
    test(s"Test Dot compilation on $name") {
      SpinalConfig().generateVerilog(DotTestComp(dt()))
    }
  }
}