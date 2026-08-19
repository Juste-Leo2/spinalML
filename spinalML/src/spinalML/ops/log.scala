package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class LogOp[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  base: Double = Math.E
) extends Component {
  val bitWidth = dataType.getBitsWidth

  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  // Domain: log(x) is undefined for x <= 0 -> return 0.0 (industry convention, same as rsqrt)
  val mathFn = (x: Double) => if (x <= 0) 0.0 else Math.log(x) / Math.log(base)

  if (bitWidth <= 8) {
    val isFloat = dataType().isInstanceOf[FloatML]
    val (valFn, encodeFn) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (MathLUTs.floatValFn(f.expBits, f.mantBits), MathLUTs.floatEncodeFn(f.expBits, f.mantBits))
    } else {
      (MathLUTs.intValFn(bitWidth), MathLUTs.intEncodeFn(bitWidth))
    }

    val lutOp = UnaryLUTOp(dataType, shape, lanes, valFn, encodeFn, mathFn)
    lutOp.io.a <> io.a
    io.c <> lutOp.io.c

  } else if (dataType().isInstanceOf[FloatML]) {
    // Algebraic separation for FloatML > 8 bits (e.g. BF16, FP16)
    // log_b(x) = log2(x) * ln(2)/ln(b)
    //  - log2(x) = (exponent - bias) + log2(1 + mantissa/2^mantBits), the fractional part
    //    is looked up in a small ROM (log2 of the mantissa fraction, Q8.8).
    //  - the fixed-point value is then multiplied by ln(2)/ln(b) in Q0.16.
    //  - the resulting real (signed) is re-quantized to FloatML via LZD normalization.
    val fType = dataType().asInstanceOf[FloatML]
    val expBits = fType.expBits
    val mantBits = fType.mantBits
    val bias = fType.bias

    // ln(2)/ln(b) in Q0.16 (e.g. base=e: 45426, base=10: 19728)
    val log2ToBase = Math.round(Math.log(2.0) / Math.log(base) * 65536.0).toInt

    // ROM: index = mantissa (mantBits), output = log2(1 + m/2^mantBits) as Q8.8 fraction
    val log2MantLuts = for (i <- 0 until lanes) yield {
      val romContent = for (m <- 0 until (1 << mantBits)) yield {
        val frac = Math.log(1.0 + m.toDouble / (1 << mantBits)) / Math.log(2.0)
        val encoded = Math.round(frac * 256.0).toInt
        B(encoded, 8 bits)
      }
      Mem(Bits(8 bits), initialContent = romContent)
    }

    val outPayload = Vec(dataType, lanes)
    val stage1_valid = RegInit(False)

    when(io.a.stream.ready) {
      stage1_valid := io.a.stream.valid
    }

    for (i <- 0 until lanes) {
      val x = io.a.stream.payload(i).asInstanceOf[FloatML]
      val isSubOrZero = x.exponent === 0 || x.sign // log(0, subnormal or x<=0) -> 0

      // 1. log2(x) in fixed point Q8.8: int part = exponent - bias, frac part = ROM
      val expTrueSInt = x.exponent.intoSInt - S(bias, expBits + 2 bits)
      val fracBits = log2MantLuts(i).readAsync(x.mantissa).asUInt
      val log2Fixed = (((expTrueSInt << 8).asUInt) | fracBits.resize(expTrueSInt.getWidth + 8)).intoSInt

      // 2. Multiply by ln(2)/ln(b) in Q0.16: Q8.8 * Q0.16 = Q8.24
      val yFixedFull = log2Fixed * S(log2ToBase, 18 bits)

      // 3. Re-quantize the resulting real (signed, Q8.24) into FloatML via LZD
      val Q = 24
      val W = 32
      val v = yFixedFull.resize(W bits)
      val absV = Mux(v < 0, -v, v)
      val isZero = absV === 0
      val reversed = absV.asBits.reversed
      val lz = spinal.lib.OHToUInt(spinal.lib.OHMasking.first(reversed))

      val posSInt = S(W - 1, lz.getWidth + 2 bits) - lz.intoSInt
      val expSInt = S(bias, expBits + 4 bits) + (posSInt - S(Q, expBits + 4 bits))

      val aligned = (absV << lz).resize(W bits)
      val mantissa = aligned(W - 2 downto W - 1 - mantBits)

      val sign = Mux(v < 0, True, False)

      val regIsZero = RegNextWhen(isZero || isSubOrZero, io.a.stream.ready)
      val expOverflow = RegNextWhen(expSInt >= S((1 << expBits) - 1, expBits + 4 bits), io.a.stream.ready)
      val expUnderflow = RegNextWhen(expSInt <= 0, io.a.stream.ready)
      val regNewExp = RegNextWhen(expSInt.asUInt.resize(expBits), io.a.stream.ready)
      val regMant = RegNextWhen(mantissa.asUInt, io.a.stream.ready)
      val regSign = RegNextWhen(sign, io.a.stream.ready)

      val outX = FloatML(expBits, mantBits)

      outX.sign := regSign
      when(regIsZero) {
        outX.sign := False
        outX.exponent := 0
        outX.mantissa := 0
      } elsewhen (expUnderflow) {
        outX.sign := False
        outX.exponent := 0
        outX.mantissa := 0
      } elsewhen (expOverflow) {
        outX.exponent := ((1 << expBits) - 1)
        outX.mantissa := 0
      } otherwise {
        outX.exponent := regNewExp
        outX.mantissa := regMant
      }

      outPayload(i).assignFrom(outX.asInstanceOf[T])
    }

    io.a.stream.ready := io.c.stream.ready || !stage1_valid
    io.c.stream.valid := stage1_valid
    io.c.stream.payload := outPayload

  } else {
    // PWL Approximation for Int > 8 bits
    val indexBits = 8
    val numSegments = 1 << indexBits

    val segmentIndexFn: T => UInt = (x: T) => {
      x.asBits(bitWidth - 1 downto bitWidth - indexBits).asUInt
    }

    val segmentFn = spinalML.utils.PWLLUTs.createSegmentFn(bitWidth, false, 0, 0, indexBits, mathFn)

    val pwlOp = spinalML.utils.UnaryPWLOp(dataType, shape, lanes, numSegments, segmentIndexFn, segmentFn)
    pwlOp.io.a <> io.a
    io.c <> pwlOp.io.c
  }
}

object log {
  def apply[T <: Data](a: Tensor[T], base: Double = Math.E): Tensor[T] = {
    val comp = LogOp(a.dataType, a.shape, a.lanes, base)
    comp.io.a <> a
    comp.io.c
  }
}