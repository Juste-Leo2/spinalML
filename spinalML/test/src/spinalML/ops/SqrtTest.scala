// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}

case class SqrtTestComp[T <: Data](dataType: HardType[T], forceAlg: Boolean = false) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> sqrt(io.a, forceAlg)
}

class SqrtTest extends AnyFunSuite {
  // OPS-02: a negative input saturates to +0 (no NaN in the fabric).
  // Bounded waits: a stall becomes a clean failure, never a hang.
  test("OPS-02 Sqrt of negative I8 saturates to 0 (LUT path)") {
    SimConfig.withWave.compile(SqrtTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0) #= -9
      dut.io.a.stream.payload(1) #= -9
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for negative input")
      assert(dut.io.c.stream.payload(0).toInt == 0,
        s"expected +0, got ${dut.io.c.stream.payload(0).toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  test("OPS-02 Sqrt of negative BF16 saturates to +0 (algebraic path)") {
    SimConfig.withWave.compile(SqrtTestComp(BF16())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      // -2.0 BF16: sign=1, exp=128, mant=0
      for (lane <- 0 until 2) {
        dut.io.a.stream.payload(lane).sign #= true
        dut.io.a.stream.payload(lane).exponent #= 128
        dut.io.a.stream.payload(lane).mantissa #= 0
      }
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for negative input")
      val s = dut.io.c.stream.payload(0).sign.toBoolean
      val e = dut.io.c.stream.payload(0).exponent.toInt
      val m = dut.io.c.stream.payload(0).mantissa.toInt
      assert(!s && e == 0 && m == 0, s"expected +0, got sign=$s exp=$e mant=$m")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Sqrt compilation on I8") { SpinalConfig().generateVerilog(SqrtTestComp(I8())) }
  test("Sqrt compilation on I16") { SpinalConfig().generateVerilog(SqrtTestComp(I16())) }
  test("Sqrt compilation on FP8") { SpinalConfig().generateVerilog(SqrtTestComp(FP8_E4M3())) }
  test("Sqrt compilation on BF16") { SpinalConfig().generateVerilog(SqrtTestComp(BF16())) }
}
