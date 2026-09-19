// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, BF16}

case class RsqrtTestComp[T <: Data](dataType: HardType[T], forceAlg: Boolean = false) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> rsqrt(io.a, forceAlg)
}

class RsqrtTest extends AnyFunSuite {
  // OPS-02: a negative input saturates to +0 (no NaN in the fabric).
  test("OPS-02 Rsqrt of negative I8 saturates to 0 (LUT path)") {
    SimConfig.withWave.compile(RsqrtTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()

      dut.io.a.stream.payload(0) #= -1
      dut.io.a.stream.payload(1) #= -1
      dut.io.a.stream.valid #= true
      assert(!dut.clockDomain.waitSamplingWhere(20)(dut.io.c.stream.valid.toBoolean),
        "no output for negative input")
      assert(dut.io.c.stream.payload(0).toInt == 0,
        s"expected +0, got ${dut.io.c.stream.payload(0).toInt}")
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  test("OPS-02 Rsqrt of negative BF16 saturates to +0 (algebraic path)") {
    SimConfig.withWave.compile(RsqrtTestComp(BF16())).doSim { dut =>
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

  test("Rsqrt LUT compilation on I8") {
    SpinalConfig().generateVerilog(RsqrtTestComp(I8()))
  }

  test("Rsqrt LUT compilation on FP8") {
    SpinalConfig().generateVerilog(RsqrtTestComp(FP8_E4M3()))
  }

  test("Rsqrt PWL compilation on I16") {
    SpinalConfig().generateVerilog(RsqrtTestComp(spinalML.dtypes.I16()))
  }

  test("Rsqrt PWL compilation on BF16") {
    SpinalConfig().generateVerilog(RsqrtTestComp(spinalML.dtypes.BF16()))
  }
}
