// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

case class FlattenTestCompFormal[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 3, 4), lanes = 2))
    val c = master(Tensor(dataType, Seq(24), lanes = 2))
  }
  io.c <> spinalML.ops.flatten(io.a)
}

class FlattenFormal extends Component {
  val dut = FormalDut(new FlattenTestCompFormal(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val lanes = dut.io.a.stream.payload.length
  val expected = Vec(I8(), lanes)
  for (lane <- 0 until lanes) {
    expected(lane) := dut.io.a.stream.payload(lane)
  }

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    for (lane <- 0 until lanes) {
      assert(dut.io.c.stream.payload(lane) === expected(lane), s"Flatten lane $lane mismatch")
    }
  }
}

object FlattenFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new FlattenFormal, "flatten_i8")
  }
}
