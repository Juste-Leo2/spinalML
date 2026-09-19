// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, U8, FP8_E4M3, I16, BF16}

case class SigmoidTestComp[T <: Data](
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
  io.c <> sigmoid(io.a, inputScale, inputZeroPoint)
}

class SigmoidTest extends AnyFunSuite {

  /**
   * Quantized (TFLite-convention) logistic: out scale 1/256, int8 zp -128 /
   * uint8 zp 0. Drives two input codes per beat (one per lane) and compares
   * the exact output codes.
   */
  private def runSigmoidCases[T <: Data](
    dataType: HardType[T],
    cases: Seq[(Int, Int)],
    expected: Seq[(Int, Int)],
    inputScale: Double = 1.0,
    inputZeroPoint: Int = 0
  ): Unit = {
    // The Python runner picks `SigmoidTestComp.v` from the test sandbox: a
    // scale/zp variant gets a distinct definition name so it cannot overwrite
    // the default-generated ROM. Construction stays inside the by-name
    // `SimConfig.compile` argument (elaboration context).
    val scaleVariant = inputScale != 1.0 || inputZeroPoint != 0
    SimConfig.compile(SigmoidTestComp(dataType, inputScale, inputZeroPoint,
      if (scaleVariant) "SigmoidScaleTestComp" else null)).doSim { dut =>
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
      assert(got.length == expected.length, s"Sigmoid: collected ${got.length}/${expected.length}")
      assert(got.toSeq == expected, s"Sigmoid: got $got expected $expected")
    }
  }

  test("Sigmoid quantized I8 codes (TFLite logistic, RNE)") {
    runSigmoidCases(
      I8(),
      cases = Seq((2, -2), (0, 10), (-10, 127)),
      expected = Seq((97, -97), (0, 127), (-128, 127))
    )
  }

  test("Sigmoid quantized U8 codes (TFLite logistic, RNE)") {
    runSigmoidCases(
      U8(),
      cases = Seq((2, 254), (0, 10), (128, 0)),
      expected = Seq((225, 255), (128, 255), (255, 128))
    )
  }

  test("Sigmoid I8 input quantization drives the LUT (scale and zero point)") {
    // x = (q - zp) * scale: q=4, scale=0.5 -> x=2 -> 97
    runSigmoidCases(
      I8(),
      cases = Seq((4, 0), (0, -4)),
      expected = Seq((97, 0), (0, -97)),
      inputScale = 0.5
    )
    // q=6, zp=4 -> x=2 -> 97 ; q=2, zp=4 -> x=-2 -> -97
    runSigmoidCases(
      I8(),
      cases = Seq((6, 4), (2, 4)),
      expected = Seq((97, 0), (-97, 0)),
      inputZeroPoint = 4
    )
  }

  test("OPS-03 Sigmoid I16 is still refused (int8 conventions only)") {
    assertThrows[IllegalArgumentException] { spinal.core.SpinalVerilog(SigmoidTestComp(I16())) }
  }

  // Compilation entry points used by the Python runner (`run_mill` selects by
  // test-name substring via `-z <dtype>`).
  test("Sigmoid quantized I8 compilation") { SpinalConfig().generateVerilog(SigmoidTestComp(I8())) }
  test("Sigmoid quantized U8 compilation") { SpinalConfig().generateVerilog(SigmoidTestComp(U8())) }
  test("Sigmoid compilation on FP8") { SpinalConfig().generateVerilog(SigmoidTestComp(FP8_E4M3())) }
  test("Sigmoid compilation on BF16") { SpinalConfig().generateVerilog(SigmoidTestComp(BF16())) }
}
