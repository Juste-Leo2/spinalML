// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

/**
 * Exact integer division helpers, mirroring ONNX `Div` semantics:
 * truncating division (rounding toward zero) on the same integer dtype, with
 * the engine's no-wrap policy on the two unreachable-by-math cases:
 *   - divisor == 0 : saturate (+max for a > 0, -min... i.e. min for a < 0, 0 for a == 0)
 *   - INT_MIN / -1 : saturates to +max (the true quotient 2^(w-1) is not
 *     representable as a positive signed value).
 * Unsigned division is ordinary floor division and cannot overflow.
 *
 * Implemented as an unrolled restoring divider for widths <= 8 (combinational,
 * 0 extra cycle). The >8-bit iterative version (Wave 5 step B) must keep this
 * exact contract.
 */
object IntDiv {

  /** Two's-complement magnitude of `x` as an unsigned w-bit value (INT_MIN -> 2^(w-1)). */
  private def absBits(x: SInt): UInt = {
    val w = x.getWidth
    val bits = x.asBits.asUInt
    val neg = ((~bits).resize(w) + U(1, w bits)).resize(w)
    Mux(x.msb, neg, bits)
  }

  /** Restoring long division on magnitudes: returns (quotient, remainder), both w bits. */
  private def unsignedCore(aAbs: UInt, bAbs: UInt, w: Int): (UInt, UInt) = {
    val quo = Bits(w bits)
    var rem = U(0, w bits)
    for (i <- (w - 1) downto 0) {
      val remShift = ((rem << 1).resize(w + 1) | aAbs(i).asUInt.resize(w + 1)).resize(w + 1)
      val ge = remShift >= bAbs.resize(w + 1)
      rem = Mux(ge, remShift - bAbs.resize(w + 1), remShift).resize(w)
      quo(i) := ge
    }
    (quo.asUInt, rem)
  }

  /** Signed exact division, truncation toward zero, saturating (ONNX `Div`). */
  def signed(a: SInt, b: SInt): SInt = {
    require(a.getWidth == b.getWidth,
      s"IntDiv.signed expects equal widths (ONNX Div same-type contract), got ${a.getWidth} and ${b.getWidth}")
    val w = a.getWidth
    val bAbs = absBits(b)
    val (quo, _) = unsignedCore(absBits(a), bAbs, w)

    val maxVal = (1 << (w - 1)) - 1
    val minVal = -(1 << (w - 1))
    val half = U(BigInt(1) << (w - 1), w bits)
    val sign = a.msb ^ b.msb

    Mux(bAbs === 0,
      // 0 divisor: signed saturation of the mathematical infinity.
      Mux(a.msb, S(minVal, w bits), Mux(a === 0, S(0, w bits), S(maxVal, w bits))),
      Mux(sign,
        // |a| == 2^(w-1) with b == ±1 lands exactly on INT_MIN (representable).
        Mux(quo === half, S(minVal, w bits), (S(0, w bits) - quo.asSInt).resize(w)),
        // |a| == 2^(w-1) with b == -1 overflows the positive range (INT_MIN/-1).
        Mux(quo === half, S(maxVal, w bits), quo.asSInt.resize(w))))
  }

  /** Unsigned exact division (floor), saturated only on 0 divisor. */
  def unsigned(a: UInt, b: UInt): UInt = {
    require(a.getWidth == b.getWidth,
      s"IntDiv.unsigned expects equal widths (ONNX Div same-type contract), got ${a.getWidth} and ${b.getWidth}")
    val w = a.getWidth
    val (quo, _) = unsignedCore(a, b, w)
    val maxVal = (BigInt(1) << w) - 1
    Mux(b === 0, Mux(a === 0, U(0, w bits), U(maxVal, w bits)), quo)
  }
}

/**
 * DivOp: element-wise `a / b`.
 *
 * FloatML keeps the historical datapath: synchronize, reciprocal of B, then a
 * multiply (bit-exact reference for the FP goldens). Integer types use the
 * exact ONNX `Div` semantics from [[IntDiv]] (truncation toward zero, no
 * wrap); the <=8-bit divider is combinational, the >8-bit iterative divider is
 * Wave 5 step B.
 */
case class DivOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  dataType() match {
    case _: FloatML => floatDatapath()
    case s: SInt =>
      require(s.getWidth <= 8,
        s"DivOp integer division is unimplemented above 8 bits (${s.getWidth}-bit): the iterative " +
        "divider is Wave 5 step B; see docs/rounding_policy.md section 7.")
      intDatapath()
    case u: UInt =>
      require(u.getWidth <= 8,
        s"DivOp integer division is unimplemented above 8 bits (${u.getWidth}-bit): the iterative " +
        "divider is Wave 5 step B; see docs/rounding_policy.md section 7.")
      intDatapath()
    case _ => throw new Exception("Unsupported data type for Div")
  }

  private def floatDatapath(): Unit = {
    // 1. Synchronize inputs
    val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
    val (syncStreamForB, syncStreamForA) = StreamFork2(syncStream)

    // 2. Calculate Reciprocal of B
    val invBComp = ReciprocalOp(dataType, shape, lanes)
    invBComp.io.a.stream << syncStreamForB.translateWith(io.b.stream.payload)

    // 3. Pipeline A to match the 1-cycle latency of ReciprocalOp
    val pipelinedA = syncStreamForA.translateWith(io.a.stream.payload).m2sPipe()
    val invBStream = invBComp.io.c.stream

    // 4. Join streams at output (safety net, they should arrive together)
    val joined = StreamJoin.arg(pipelinedA, invBStream)

    val outValid = RegInit(False)
    when(io.c.stream.ready || !outValid) {
      outValid := joined.valid
    }

    val outPayload = Reg(Vec(dataType, lanes))
    when(joined.valid && (io.c.stream.ready || !outValid)) {
      for (i <- 0 until lanes) {
        val pa = pipelinedA.payload(i)
        val pInvB = invBStream.payload(i)

        (pa, pInvB) match {
          case (va: SInt, vInvB: SInt) =>
            outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
          case (va: UInt, vInvB: UInt) =>
            outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
          case (va: FloatML, vInvB: FloatML) =>
            outPayload(i).assignFrom(spinalML.utils.Float.mul(va, vInvB).asInstanceOf[T])
          case _ => throw new Exception("Unsupported data type for Div")
        }
      }
    }

    joined.ready := io.c.stream.ready || !outValid
    io.c.stream.valid := outValid
    io.c.stream.payload := outPayload
  }

  private def intDatapath(): Unit = {
    val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)

    // Combinational exact divider followed by the same elastic output register
    // as the float path (breaks the divider -> downstream path, and keeps the
    // component clocked even though the arithmetic itself is combinational).
    val resPayload = Vec(dataType, lanes)
    for (i <- 0 until lanes) {
      resPayload(i) := ((io.a.stream.payload(i), io.b.stream.payload(i)) match {
        case (va: SInt, vb: SInt) => IntDiv.signed(va, vb)
        case (va: UInt, vb: UInt) => IntDiv.unsigned(va, vb)
        case _ => throw new Exception("Unsupported data type for Div")
      }).asInstanceOf[T]
    }

    val outValid = RegInit(False)
    when(io.c.stream.ready || !outValid) {
      outValid := syncStream.valid
    }

    val outPayload = Reg(Vec(dataType, lanes))
    when(syncStream.valid && (io.c.stream.ready || !outValid)) {
      outPayload := resPayload
    }

    syncStream.ready := io.c.stream.ready || !outValid
    io.c.stream.valid := outValid
    io.c.stream.payload := outPayload
  }
}

object div {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    val comp = DivOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.b <> b
    comp.io.c
  }
}
