// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.utils

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.{BF16, FP4_E2M1, FP8_E4M3}
import spinalML.utils.Float

/**
 * Netlist-level guard for NaN propagation (companion of the
 * simulation-side [[spinalML.dtypes.FloatNaNTest]] and the exhaustive
 * `FloatSweepTest` NaN sweeps).
 *
 * Deliberately oracle-free (unlike Add/Mul formals, which compare against
 * the same Scala function and are tautological at the algorithm level):
 * every property below is a direct pattern over the DUT wires.
 *
 *   - non-emission: with non-NaN inputs, no output ever carries a NaN
 *     pattern (saturation keeps yielding 448/inf);
 *   - propagation: a NaN input pattern yields the canonical NaN output
 *     (first-NaN sign rule), every comparison with NaN is False, and
 *     `max` inherits `Mux(gt, a, b)` (second operand wins on NaN).
 */
case class FloatNaNTestComp() extends Component {
  val io = new Bundle {
    val a8 = in(FP8_E4M3())
    val b8 = in(FP8_E4M3())
    val mul8 = out(FP8_E4M3())
    val add8 = out(FP8_E4M3())
    val gt8 = out(Bool())
    val max8 = out(FP8_E4M3())
    val r8 = in(FP8_E4M3())
    val rt8 = out(FP8_E4M3())
    val w8 = out(BF16())
    val bf = in(BF16())
    val rb = out(FP8_E4M3())
    val a4 = in(FP4_E2M1())
    val b4 = in(FP4_E2M1())
    val mul4 = out(FP4_E2M1())
  }
  io.mul8 := Float.mul(io.a8, io.b8)
  io.add8 := Float.add(io.a8, io.b8)
  io.gt8 := Float.gt(io.a8, io.b8)
  io.max8 := Float.max(io.a8, io.b8)
  io.rt8 := Float.roundTo(io.r8, 4, 3)
  io.w8 := Float.widen(io.r8, 8, 7)
  io.rb := Float.roundTo(io.bf, 4, 3)
  io.mul4 := Float.mul(io.a4, io.b4)
}

class FloatNaNFormal extends Component {
  val dut = FormalDut(FloatNaNTestComp())

  anyseq(dut.io.a8)
  anyseq(dut.io.b8)
  anyseq(dut.io.r8)
  anyseq(dut.io.bf)
  anyseq(dut.io.a4)
  anyseq(dut.io.b4)

  assumeInitial(clockDomain.isResetActive)

  // NaN input patterns (E4M3 single slot; FP4 (3, 1); BF16 all-ones + mantissa).
  val a8NaN = dut.io.a8.exponent === U(15, 4 bits) && dut.io.a8.mantissa === U(7, 3 bits)
  val b8NaN = dut.io.b8.exponent === U(15, 4 bits) && dut.io.b8.mantissa === U(7, 3 bits)
  val r8NaN = dut.io.r8.exponent === U(15, 4 bits) && dut.io.r8.mantissa === U(7, 3 bits)
  val bfNaN = dut.io.bf.exponent === U(255, 8 bits) && dut.io.bf.mantissa =/= U(0, 7 bits)
  val a4NaN = dut.io.a4.exponent === U(3, 2 bits) && dut.io.a4.mantissa === U(1, 1 bits)
  val b4NaN = dut.io.b4.exponent === U(3, 2 bits) && dut.io.b4.mantissa === U(1, 1 bits)

  // NaN output patterns.
  val mul8NaN = dut.io.mul8.exponent === U(15, 4 bits) && dut.io.mul8.mantissa === U(7, 3 bits)
  val add8NaN = dut.io.add8.exponent === U(15, 4 bits) && dut.io.add8.mantissa === U(7, 3 bits)
  val max8NaN = dut.io.max8.exponent === U(15, 4 bits) && dut.io.max8.mantissa === U(7, 3 bits)
  val rt8NaN = dut.io.rt8.exponent === U(15, 4 bits) && dut.io.rt8.mantissa === U(7, 3 bits)
  val w8NaN = dut.io.w8.exponent === U(255, 8 bits) && dut.io.w8.mantissa === U(1, 7 bits)
  val rbNaN = dut.io.rb.exponent === U(15, 4 bits) && dut.io.rb.mantissa === U(7, 3 bits)
  val mul4NaN = dut.io.mul4.exponent === U(3, 2 bits) && dut.io.mul4.mantissa === U(1, 1 bits)

  // 1. Non-emission: finite inputs never produce a NaN pattern.
  when(!a8NaN && !b8NaN) {
    assert(!mul8NaN, "mul emitted NaN from finite inputs")
    assert(!add8NaN, "add emitted NaN from finite inputs")
    assert(!max8NaN, "max emitted NaN from finite inputs")
  }
  when(!r8NaN) {
    assert(!rt8NaN, "roundTo emitted NaN from finite input")
    assert(!w8NaN, "widen emitted NaN from finite input")
  }
  when(!bfNaN) {
    assert(!rbNaN, "roundTo emitted NaN from finite BF16 input")
  }
  when(!a4NaN && !b4NaN) {
    assert(!mul4NaN, "FP4 mul emitted NaN from finite inputs")
  }

  // 2. Propagation: NaN in -> canonical NaN out, first-NaN sign rule.
  when(a8NaN) {
    assert(mul8NaN && dut.io.mul8.sign === dut.io.a8.sign, "mul did not propagate a-NaN")
    assert(add8NaN && dut.io.add8.sign === dut.io.a8.sign, "add did not propagate a-NaN")
  }
  when(!a8NaN && b8NaN) {
    assert(mul8NaN && dut.io.mul8.sign === dut.io.b8.sign, "mul did not propagate b-NaN")
    assert(add8NaN && dut.io.add8.sign === dut.io.b8.sign, "add did not propagate b-NaN")
  }
  when(a8NaN || b8NaN) {
    assert(!dut.io.gt8, "gt with NaN must be False")
    assert(dut.io.max8.asBits === dut.io.b8.asBits, "max must return b on NaN")
  }
  when(r8NaN) {
    assert(rt8NaN && dut.io.rt8.sign === dut.io.r8.sign, "roundTo did not propagate NaN")
    assert(w8NaN && dut.io.w8.sign === dut.io.r8.sign, "widen did not propagate NaN")
  }
  when(bfNaN) {
    assert(rbNaN, "roundTo did not fold foreign NaN to the E4M3 slot")
  }
  when(a4NaN) {
    assert(mul4NaN && dut.io.mul4.sign === dut.io.a4.sign, "FP4 mul did not propagate a-NaN")
  }
  when(!a4NaN && b4NaN) {
    assert(mul4NaN && dut.io.mul4.sign === dut.io.b4.sign, "FP4 mul did not propagate b-NaN")
  }
}

object FloatNaNFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new FloatNaNFormal, "floatnan_formal")
  }
}
