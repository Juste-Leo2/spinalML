// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

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

// OPS-04: 2D geometry, slice rows [1,3) of [4,3] at lanes=1 ->
// elements 4..9 (6 beats out of 12 in).
case class SliceTest2D[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4, 3), lanes = 1))
    val c = master(Tensor(dataType, Seq(2, 3), lanes = 1))
  }

  // Keep rows 1 and 2, drop rows 0 and 3
  io.c <> spinalML.ops.slice(io.a, start = 1, end = 3, axis = 0)
}

class SliceTest extends AnyFunSuite {
  // OPS-04 red test: start/end are axis-0 row indices, i.e. beat ranges
  // [start*beatsPerRow, end*beatsPerRow), not raw beat indices.
  test("OPS-04 Slice axis0 2D lanes=1 keeps rows [1,3) = elements 4..9") {
    SimConfig.withWave.compile(SliceTest2D(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // NOTE: the op has no output buffer (c.valid is combinational on the
      // input), so forwarded beats are sampled as each input is accepted.
      dut.io.a.stream.valid #= true
      var got = Vector.empty[Int]
      for (v <- 1 to 12) {
        dut.io.a.stream.payload(0) #= v
        // Bounded: the drop path is always ready, the forward path needs
        // the output ready (held true), so every beat must be accepted.
        assert(!dut.clockDomain.waitSamplingWhere(100)(dut.io.a.stream.ready.toBoolean),
          s"input beat $v never accepted")
        if (dut.io.c.stream.valid.toBoolean) got = got :+ dut.io.c.stream.payload(0).toInt
      }
      dut.io.a.stream.valid #= false

      val expected = Seq(4, 5, 6, 7, 8, 9)
      assert(got == expected, s"expected $expected, got $got")
      dut.clockDomain.waitSampling(5)
    }
  }


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
