// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, I16, FP8_E4M3, BF16}

case class ScaleAddTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(2), lanes = 2))
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> scale_add(io.x, io.a, io.b)
}

class ScaleAddTest extends AnyFunSuite {
  test("ScaleAdd simulation on I8") {
    SimConfig.withWave.compile(ScaleAddTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.x.stream.valid #= false
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      dut.io.x.stream.valid #= true
      dut.io.x.stream.payload(0) #= 2
      dut.io.x.stream.payload(1) #= -2
      
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= 2
      dut.io.a.stream.payload(1) #= 3
      
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 1
      dut.io.b.stream.payload(1) #= 1
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      assert(dut.io.c.stream.payload(0).toInt == 5)
      assert(dut.io.c.stream.payload(1).toInt == -5)
      
      dut.io.x.stream.valid #= false
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  test("ScaleAdd compilation on I8") { SpinalConfig().generateVerilog(ScaleAddTestComp(I8())) }
  test("ScaleAdd compilation on I16") { SpinalConfig().generateVerilog(ScaleAddTestComp(I16())) }
  test("ScaleAdd compilation on FP8") { SpinalConfig().generateVerilog(ScaleAddTestComp(FP8_E4M3())) }
  test("ScaleAdd compilation on BF16") { SpinalConfig().generateVerilog(ScaleAddTestComp(BF16())) }
}
