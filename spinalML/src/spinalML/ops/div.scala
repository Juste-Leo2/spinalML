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
 * Widths <= 8 use the unrolled restoring divider below (combinational). Widths
 * > 8 use `DivOp`'s serial restoring FSM, which reuses `absBits` and the same
 * saturation policy so both paths are bit-identical.
 */
object IntDiv {

  /** Two's-complement magnitude of `x` as an unsigned w-bit value (INT_MIN -> 2^(w-1)). */
  def absBits(x: SInt): UInt = {
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
 * wrap): widths <= 8 are combinational (unrolled restoring divider), widths
 * > 8 use a serial restoring FSM with a fixed `width + 2` cycle latency (one
 * beat in flight, upstream backpressure through `ready`). Both integer paths
 * are bit-identical by construction.
 */
case class DivOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  dataType() match {
    case _: FloatML => floatDatapath()
    case _: SInt | _: UInt => intDatapath()
    case _ => throw new Exception("Unsupported data type for Div")
  }

  private def intDatapath(): Unit = {
    val width = dataType.getBitsWidth
    require(width <= 32,
      s"DivOp integer division is unimplemented above 32 bits (${width}-bit): the serial divider " +
      "covers every current SInt/UInt dtype (I4/I8/U4/U8/I16/I32).")
    if (width <= 8) combIntDatapath() else serialIntDatapath(width)
  }

  private def floatDatapath(): Unit = {
    val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
    val (syncStreamForB, syncStreamForA) = StreamFork2(syncStream)

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

  private def combIntDatapath(): Unit = {
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

  /**
   * Serial restoring divider for widths > 8: one beat in flight, `width`
   * iterations plus the latch/emit cycles (`width + 2` cycles latency). The
   * per-lane datapath mirrors [[IntDiv]] bit-for-bit: magnitudes, sign XOR,
   * truncation toward zero, div0 and INT_MIN/-1 saturation.
   */
  private def serialIntDatapath(width: Int): Unit = {
    val isSigned = dataType().isInstanceOf[SInt]
    val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)

    val busy = RegInit(False)
    val done = RegInit(False)
    val cnt = Reg(UInt(log2Up(width) bits)) init (0)

    val aShift = Vec(Reg(UInt(width bits)), lanes)
    val bAbs = Vec(Reg(UInt(width bits)), lanes)
    val rem = Vec(Reg(UInt(width bits)), lanes)
    val quo = Vec(Reg(UInt(width bits)), lanes)
    val aNeg = Vec(Reg(Bool), lanes)
    val aZero = Vec(Reg(Bool), lanes)
    val bZero = Vec(Reg(Bool), lanes)
    val sign = Vec(Reg(Bool), lanes)

    syncStream.ready := !busy && !done

    when(syncStream.fire) {
      busy := True
      cnt := 0
      for (i <- 0 until lanes) {
        (io.a.stream.payload(i), io.b.stream.payload(i)) match {
          case (x: SInt, y: SInt) =>
            aShift(i) := IntDiv.absBits(x)
            bAbs(i) := IntDiv.absBits(y)
            aNeg(i) := x.msb
            aZero(i) := x === 0
            bZero(i) := y === 0
            sign(i) := x.msb ^ y.msb
          case (x: UInt, y: UInt) =>
            aShift(i) := x
            bAbs(i) := y
            aNeg(i) := False
            aZero(i) := x === 0
            bZero(i) := y === 0
            sign(i) := False
          case _ => throw new Exception("Unsupported data type for Div")
        }
        rem(i) := 0
        quo(i) := 0
      }
    }

    when(busy) {
      val last = cnt === U(width - 1, cnt.getWidth bits)
      for (i <- 0 until lanes) {
        val bit = aShift(i)(width - 1)
        val remShift = ((rem(i) << 1).resize(width + 1) | bit.asUInt.resize(width + 1)).resize(width + 1)
        val ge = remShift >= bAbs(i).resize(width + 1)
        rem(i) := Mux(ge, (remShift - bAbs(i).resize(width + 1)).resize(width), remShift.resize(width))
        val shiftedQuo = (quo(i) << 1).resize(width)
        quo(i) := Mux(ge, shiftedQuo | U(1, width bits), shiftedQuo)
        aShift(i) := (aShift(i) << 1).resize(width)
      }
      when(last) {
        busy := False
        done := True
      } otherwise {
        cnt := cnt + 1
      }
    }

    val outPayload = Vec(dataType, lanes)
    for (i <- 0 until lanes) {
      if (isSigned) {
        val maxVal = (1 << (width - 1)) - 1
        val minVal = -(1 << (width - 1))
        val half = U(BigInt(1) << (width - 1), width bits)
        val res = Mux(bZero(i),
          Mux(aNeg(i), S(minVal, width bits), Mux(aZero(i), S(0, width bits), S(maxVal, width bits))),
          Mux(sign(i),
            Mux(quo(i) === half, S(minVal, width bits), (S(0, width bits) - quo(i).asSInt).resize(width)),
            Mux(quo(i) === half, S(maxVal, width bits), quo(i).asSInt)))
        outPayload(i).assignFrom(res.asInstanceOf[T])
      } else {
        val maxVal = (BigInt(1) << width) - 1
        val res = Mux(bZero(i), Mux(aZero(i), U(0, width bits), U(maxVal, width bits)), quo(i))
        outPayload(i).assignFrom(res.asInstanceOf[T])
      }
    }

    io.c.stream.valid := done
    io.c.stream.payload := outPayload
    when(done && io.c.stream.ready) {
      done := False
    }
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
