// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.{RoundingConfig, RoundingMode}
import spinalML.tensors.Tensor
import spinalML.dtypes.{I32, I8}
import org.scalatest.funsuite.AnyFunSuite

/** Requantize topologies parameterized by rounding mode. */
case class RequantizeRoundingTestComp(shift: Int, rounding: RoundingMode) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I32(), Seq(4), lanes = 4))
    val c = master(Tensor(I8(), Seq(4), lanes = 4))
  }
  io.c <> requantize(io.a, I8(), shift, rounding)
}

class RoundingPolicyTest extends AnyFunSuite {

  test("RoundingConfig defaults to RNE and parses trunc aliases") {
    assert(RoundingConfig.fromString("rne") == RoundingMode.Rne)
    assert(RoundingConfig.fromString("anything-unknown") == RoundingMode.Rne)
    assert(RoundingConfig.fromString("trunc") == RoundingMode.Truncate)
    assert(RoundingConfig.fromString("truncate") == RoundingMode.Truncate)
    assert(RoundingConfig.fromString("floor") == RoundingMode.Truncate)
    assert(RoundingConfig.fromString("TRUNC") == RoundingMode.Truncate)
  }

  private def runRequantize(
    dut: RequantizeRoundingTestComp,
    inputs: Seq[Seq[Int]]
  ): Seq[Seq[Int]] = {
    dut.clockDomain.forkStimulus(period = 10)
    StreamReadyRandomizer(dut.io.c.stream, dut.clockDomain)
    dut.io.a.stream.valid #= false
    dut.clockDomain.waitSampling(5)

    var outputs = Vector.empty[Seq[Int]]
    fork {
      var remaining = inputs.length
      while (remaining > 0) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          outputs = outputs :+ (0 until 4).map(i =>
            dut.io.c.stream.payload(i).asInstanceOf[SInt].toInt
          )
          remaining -= 1
        }
      }
    }
    for (row <- inputs) {
      dut.io.a.stream.valid #= true
      for (lane <- 0 until 4) {
        dut.io.a.stream.payload(lane).asInstanceOf[SInt] #= row(lane)
      }
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(scala.util.Random.nextInt(5))
    }
    dut.clockDomain.waitSamplingWhere(outputs.length == inputs.length)
    dut.clockDomain.waitSampling(5)
    outputs
  }

  /** RNE golden (Scala-side, integer shift): ties to even, then saturate to I8. */
  private def rneShift(x: Int, shift: Int): Int = {
    if (shift <= 0) {
      val t = x
      return t.max(-128).min(127)
    }
    val truncated = Math.floorDiv(x, 1 << shift)
    val mask = (1 << shift) - 1
    // Exact remainder in [0, 2^shift): works for negatives via floorDiv identity.
    val rem = x - truncated * (1 << shift)
    val half = 1 << (shift - 1)
    val roundUp = if (rem > half) true
    else if (rem < half) false
    else (truncated & 1) != 0 // tie -> even
    val rounded = if (roundUp) truncated + 1 else truncated
    rounded.max(-128).min(127)
  }

  test("Requantize RNE rounds ties to even (shift=1)") {
    SimConfig.compile(RequantizeRoundingTestComp(shift = 1, rounding = RoundingMode.Rne)).doSim { dut =>
      // shift=1: x=1 -> 0.5 -> 0, x=3 -> 1.5 -> 2, x=-1 -> -0.5 -> 0, x=-3 -> -1.5 -> -2
      val out = runRequantize(dut, Seq(Seq(1, 3, -1, -3)))
      assert(out.head == Seq(0, 2, 0, -2), s"RNE ties failed: got ${out.head}")
    }
  }

  test("Requantize Truncate stays bit-exact legacy (shift=1)") {
    SimConfig.compile(RequantizeRoundingTestComp(shift = 1, rounding = RoundingMode.Truncate)).doSim { dut =>
      // Legacy: arithmetic shift truncates toward -inf: 1->0, 3->1, -1->-1, -3->-2
      val out = runRequantize(dut, Seq(Seq(1, 3, -1, -3)))
      assert(out.head == Seq(0, 1, -1, -2), s"Truncate legacy changed: got ${out.head}")
    }
  }

  test("Requantize RNE exhaustive sweep I8 range (shift=2)") {
    SimConfig.compile(RequantizeRoundingTestComp(shift = 2, rounding = RoundingMode.Rne)).doSim { dut =>
      val rows = (0 until 256 by 4).map(base => (base until base + 4).map(v => v - 128).toSeq).toSeq
      val out = runRequantize(dut, rows)
      val flatIn = rows.flatten
      val flatOut = out.flatten
      val expected = flatIn.map(v => rneShift(v, 2))
      assert(flatOut == expected, s"RNE sweep mismatch at ${flatOut.zip(expected).indexWhere { case (a, b) => a != b }}")
    }
  }

  test("Requantize Truncate exhaustive sweep matches legacy shift (shift=2)") {
    SimConfig.compile(RequantizeRoundingTestComp(shift = 2, rounding = RoundingMode.Truncate)).doSim { dut =>
      val rows = (0 until 256 by 4).map(base => (base until base + 4).map(v => v - 128).toSeq).toSeq
      val out = runRequantize(dut, rows)
      val flatIn = rows.flatten
      val flatOut = out.flatten
      val expected = flatIn.map(v => (if (v >= 0) v >> 2 else Math.floorDiv(v, 4)).max(-128).min(127))
      assert(flatOut == expected, "Truncate sweep diverged from legacy")
    }
  }

  test("Requantize topologies elaborate in both modes") {
    SpinalConfig().generateVerilog(RequantizeRoundingTestComp(shift = 2, rounding = RoundingMode.Rne))
    SpinalConfig().generateVerilog(RequantizeRoundingTestComp(shift = 2, rounding = RoundingMode.Truncate))
  }
}
