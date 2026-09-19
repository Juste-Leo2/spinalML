// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, U8, FP8_E4M3, I16, BF16}

case class TanhTestComp[T <: Data](
  dataType: HardType[T],
  inputScale: Double = 1.0,
  inputZeroPoint: Int = 0,
  defName: String = null
) extends Component {
  if (defName != null) setDefinitionName(defName)
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2), lanes = 2))
  }
  io.c <> tanh(io.a, inputScale, inputZeroPoint)
}

class TanhTest extends AnyFunSuite {

  /**
   * Quantized (TFLite-convention) tanh: out scale 1/128, int8 zp 0 / uint8
   * zp 128. Drives two input codes per beat (one per lane) and compares the
   * exact output codes.
   */
  private def runTanhCases[T <: Data](
    dataType: HardType[T],
    cases: Seq[(Int, Int)],
    expected: Seq[(Int, Int)],
    inputScale: Double = 1.0,
    inputZeroPoint: Int = 0
  ): Unit = {
    // The Python runner picks `TanhTestComp.v` from the test sandbox: a
    // scale/zp variant gets a distinct definition name so it cannot overwrite
    // the default-generated ROM. Construction stays inside the by-name
    // `SimConfig.compile` argument (elaboration context).
    val scaleVariant = inputScale != 1.0 || inputZeroPoint != 0
    SimConfig.compile(TanhTestComp(dataType, inputScale, inputZeroPoint,
      if (scaleVariant) "TanhScaleTestComp" else null)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
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

      def drive(a0: Int, a1: Int): Unit = {
        dut.io.a.stream.payload(0).asInstanceOf[BaseType].assignBigInt(BigInt(a0))
        dut.io.a.stream.payload(1).asInstanceOf[BaseType].assignBigInt(BigInt(a1))
        dut.io.a.stream.valid #= true
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
        dut.io.a.stream.valid #= false
      }

      for ((a0, a1) <- cases) drive(a0, a1)

      var cycles = 0
      while (got.length < expected.length && cycles < 1000) {
        dut.clockDomain.waitSampling()
        cycles += 1
      }
      assert(got.length == expected.length, s"Tanh: collected ${got.length}/${expected.length}")
      assert(got.toSeq == expected, s"Tanh: got $got expected $expected")
    }
  }

  test("Tanh quantized I8 codes (TFLite tanh, RNE)") {
    runTanhCases(
      I8(),
      cases = Seq((1, -1), (0, 10), (-10, -128)),
      expected = Seq((97, -97), (0, 127), (-128, -128))
    )
  }

  test("Tanh quantized U8 codes (TFLite tanh, RNE)") {
    runTanhCases(
      U8(),
      cases = Seq((1, 255), (0, 10), (128, 0)),
      expected = Seq((225, 255), (128, 255), (255, 128))
    )
  }

  test("Tanh I8 input quantization drives the LUT (scale and zero point)") {
    // x = (q - zp) * scale: q=1, scale=2 -> x=2 -> 128*tanh(2) = 123
    runTanhCases(
      I8(),
      cases = Seq((0, 1)),
      expected = Seq((0, 123)),
      inputScale = 2.0
    )
    // q=11, zp=10 -> x=1 -> 97
    runTanhCases(
      I8(),
      cases = Seq((10, 11)),
      expected = Seq((0, 97)),
      inputZeroPoint = 10
    )
  }

  test("OPS-03 Tanh I16 is still refused (int8 conventions only)") {
    assertThrows[IllegalArgumentException] { spinal.core.SpinalVerilog(TanhTestComp(I16())) }
  }

  // Compilation entry points used by the Python runner (`run_mill` selects by
  // test-name substring via `-z <dtype>`).
  test("Tanh quantized I8 compilation") { SpinalConfig().generateVerilog(TanhTestComp(I8())) }
  test("Tanh quantized U8 compilation") { SpinalConfig().generateVerilog(TanhTestComp(U8())) }
  test("Tanh compilation on FP8") { SpinalConfig().generateVerilog(TanhTestComp(FP8_E4M3())) }
  test("Tanh compilation on BF16") { SpinalConfig().generateVerilog(TanhTestComp(BF16())) }
}
