// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, U8, FP8_E4M3, I16, BF16}

case class DivTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val b = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> div(io.a, io.b)
}

class DivTest extends AnyFunSuite {

  /**
   * Drive two (a, b) pairs per beat (one per lane) and compare the exact
   * integer quotient against the ONNX Div semantics: truncation toward zero,
   * no wrap (0 divisor and INT_MIN/-1 saturate).
   */
  private def runDivCases[T <: Data](
    dataType: HardType[T],
    cases: Seq[((Int, Int), (Int, Int))],
    expected: Seq[(Int, Int)]
  ): Unit = {
    SimConfig.compile(DivTestComp(dataType)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      val got = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
      fork {
        while (true) {
          if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
            got += ((dut.io.c.stream.payload(0).asInstanceOf[BaseType].toBigInt.toInt,
              dut.io.c.stream.payload(1).asInstanceOf[BaseType].toBigInt.toInt))
          }
          dut.clockDomain.waitSampling()
        }
      }

      def drive(a0: Int, a1: Int, b0: Int, b1: Int): Unit = {
        dut.io.a.stream.payload(0).asInstanceOf[BaseType].assignBigInt(BigInt(a0))
        dut.io.a.stream.payload(1).asInstanceOf[BaseType].assignBigInt(BigInt(a1))
        dut.io.b.stream.payload(0).asInstanceOf[BaseType].assignBigInt(BigInt(b0))
        dut.io.b.stream.payload(1).asInstanceOf[BaseType].assignBigInt(BigInt(b1))
        dut.io.a.stream.valid #= true
        dut.io.b.stream.valid #= true
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean && dut.io.b.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
        dut.io.b.stream.valid #= false
      }

      for (((a0, a1), (b0, b1)) <- cases) drive(a0, a1, b0, b1)

      var cycles = 0
      while (got.length < expected.length && cycles < 1000) {
        dut.clockDomain.waitSampling()
        cycles += 1
      }
      assert(got.length == expected.length, s"Div: collected ${got.length}/${expected.length}")
      assert(got.toSeq == expected, s"Div: got $got expected $expected")
    }
  }

  test("Div exact signed I8 truncates toward zero and saturates (ONNX semantics)") {
    runDivCases(
      I8(),
      cases = Seq(
        ((10, -7), (4, 2)),      // (2, -3)
        ((7, -7), (-2, -2)),     // (-3, 3)
        ((-128, 5), (-1, 0)),    // (127 sat, 127 sat)
        ((-128, 0), (1, 5)),     // (-128, 0)
        ((1, 100), (0, 3))       // (127 sat, 33)
      ),
      expected = Seq((2, -3), (-3, 3), (127, 127), (-128, 0), (127, 33))
    )
  }

  test("Div exact unsigned U8 is ordinary floor division") {
    runDivCases(
      U8(),
      cases = Seq(
        ((200, 255), (3, 0)),    // (66, 255 sat)
        ((0, 7), (0, 2))         // (0, 3)
      ),
      expected = Seq((66, 255), (0, 3))
    )
  }

  test("OPS-12 Div I16 integer division is rejected until the >8-bit divider (step B)") {
    assertThrows[IllegalArgumentException] { spinal.core.SpinalVerilog(DivTestComp(I16())) }
  }

  // Compilation entry points used by the Python runner (`run_mill` selects by
  // test-name substring via `-z <dtype>`).
  test("Div exact I8 compilation") { SpinalConfig().generateVerilog(DivTestComp(I8())) }
  test("Div exact U8 compilation") { SpinalConfig().generateVerilog(DivTestComp(U8())) }
  test("Div LUT compilation on FP8") { SpinalConfig().generateVerilog(DivTestComp(FP8_E4M3())) }
  test("Div PWL compilation on BF16") { SpinalConfig().generateVerilog(DivTestComp(BF16())) }
}
