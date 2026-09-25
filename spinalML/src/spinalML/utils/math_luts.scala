// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.utils

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.{RoundingConfig, RoundingMode}

object MathLUTs {
  /**
   * Elaboration-time round-to-nearest-even.
   * Bit-exact counterpart of Python `round()` / numpy / torch (half-to-even),
   * unlike `Math.round` (half-up). Intended for LUT/ROM constant generation
   * (mantissa ROMs, `intEncodeFn`).
   */
  def roundRNE(x: Double): Long =
    BigDecimal(x).setScale(0, BigDecimal.RoundingMode.HALF_EVEN).toLongExact

  def generateROM(bitWidth: Int, valFn: Int => Double, encodeFn: Double => BigInt, mathFn: Double => Double): Mem[Bits] = {
    val states = 1 << bitWidth
    val romContent = for (i <- 0 until states) yield {
      val x = valFn(i)
      val y = mathFn(x)
      val encoded = encodeFn(y)
      B(encoded, bitWidth bits)
    }
    Mem(Bits(bitWidth bits), initialContent = romContent)
  }

  // Generates a ROM specifically for FloatML mantissa fraction processing (Algebraic Separation).
  // Under [[Rne]] the output quantizes half-even via [[roundRNE]];
  // [[Truncate]] keeps the legacy `Math.round` (half-up) bit-exact.
  // Elaboration-only switch.
  def generateFloatMantissaROM(inBits: Int, outBits: Int, mathFn: Double => Double,
                               rounding: RoundingMode = RoundingConfig.current): Mem[Bits] = {
    val states = 1 << inBits
    val romContent = for (i <- 0 until states) yield {
      val mantFraction = i.toDouble / states
      val realMant = 1.0 + mantFraction
      val y = mathFn(realMant) // y should ideally be within [1.0, 2.0)
      val frac = y - 1.0
      val outStates = 1 << outBits
      val encoded = (if (rounding == RoundingMode.Rne) roundRNE(frac * outStates)
                     else Math.round(frac * outStates)).toInt
      B(if (encoded >= outStates) outStates - 1 else encoded, outBits bits)
    }
    Mem(Bits(outBits bits), initialContent = romContent)
  }

  // Integer codecs (2's complement). The clamped value rounds half-even
  // under [[Rne]] vs legacy half-up under [[Truncate]] (cf. uintEncodeFn).
  def intValFn(bitWidth: Int): Int => Double = i => {
    val maxVal = 1 << bitWidth
    val halfVal = 1 << (bitWidth - 1)
    if (i >= halfVal) (i - maxVal).toDouble else i.toDouble
  }

  /** Unsigned code -> real (identity): the code IS the value. */
  def uintValFn(bitWidth: Int): Int => Double = i => i.toDouble

  /** Real -> unsigned code with Rne/Truncate rounding, saturating to [0, 2^w-1]. */
  def uintEncodeFn(bitWidth: Int,
                   rounding: RoundingMode = RoundingConfig.current): Double => BigInt = y => {
    val maxVal = (1 << bitWidth) - 1
    val clampedD = Math.max(0.0, Math.min(maxVal.toDouble, y))
    val rounded: Long = if (rounding == RoundingMode.Rne) roundRNE(clampedD) else Math.round(clampedD)
    BigInt(rounded)
  }

  def intEncodeFn(bitWidth: Int,
                  rounding: RoundingMode = RoundingConfig.current): Double => BigInt = y => {
    val maxVal = (1 << (bitWidth - 1)) - 1
    val minVal = -(1 << (bitWidth - 1))
    // Clamp FIRST in the Double domain: segment slopes/intercepts (e.g. exp
    // on I16) can exceed Long range, and roundRNE.toLongExact throws on
    // overflow while Math.round saturates. Clamp-then-round is exactly
    // equivalent (out-of-range values saturate either way).
    val clampedD = Math.max(minVal.toDouble, Math.min(maxVal.toDouble, y))
    val rounded: Long = if (rounding == RoundingMode.Rne) roundRNE(clampedD) else Math.round(clampedD)
    val intVal = rounded.toInt
    if (intVal < 0) BigInt(intVal + (1 << bitWidth)) else BigInt(intVal)
  }

  // FloatML codecs (IEEE-like)
  def floatValFn(expBits: Int, mantBits: Int): Int => Double = i => {
    val sign = (i >> (expBits + mantBits)) & 1
    val exp = (i >> mantBits) & ((1 << expBits) - 1)
    val mant = i & ((1 << mantBits) - 1)
    
    val bias = (1 << (expBits - 1)) - 1
    if (exp == 0 && mant == 0) 0.0
    else {
      val realExp = exp - bias
      val realMant = 1.0 + mant.toDouble / (1 << mantBits)
      val v = realMant * Math.pow(2.0, realExp)
      if (sign == 1) -v else v
    }
  }
  
  def floatEncodeFn(expBits: Int, mantBits: Int,
                    rounding: RoundingMode = RoundingConfig.current): Double => BigInt = y => {
    if (y.isNaN) BigInt(0)
    else if (y.isInfinity) {
      // No infinity in E4M3: saturate to max finite (448, sign-preserved).
      val (satExp, satMant) = Float.satEncoding(expBits, mantBits)
      val sign = if (y < 0) 1 else 0
      BigInt((sign << (expBits + mantBits)) | (satExp << mantBits) | satMant)
    }
    else if (y == 0.0) BigInt(0)
    else {
      val sign = if (y < 0) 1 else 0
      val absY = Math.abs(y)
      val bias = (1 << (expBits - 1)) - 1
      
      var exp = Math.floor(Math.log(absY) / Math.log(2.0)).toInt
      var mant = (absY / Math.pow(2.0, exp)) - 1.0
      
      var expEnc = exp + bias
      // The rounding-overflow carry below is kept in both modes (ties can round up).
      var mantEnc = (if (rounding == RoundingMode.Rne) roundRNE(mant * (1 << mantBits))
                     else Math.round(mant * (1 << mantBits))).toInt
      
      if (mantEnc == (1 << mantBits)) { // Rounding overflow
         mantEnc = 0
         expEnc += 1
      }

      // E4M3 keeps finite field-15 values (256..448); only past-the-max or
      // the NaN slot (mantissa 7 at field 15) saturates to 448.
      val expMax = (1 << expBits) - 1
      val needsSat = if (Float.isE4M3(expBits, mantBits))
        expEnc > expMax || (expEnc == expMax && mantEnc == (1 << mantBits) - 1)
      else
        expEnc >= expMax
      if (needsSat) { // Overflow (saturate: 448 for E4M3, canonical infinity otherwise)
        val (satExp, satMant) = Float.satEncoding(expBits, mantBits)
        expEnc = satExp
        mantEnc = satMant
      } else if (expEnc <= 0) { // Underflow
        expEnc = 0
        mantEnc = 0
      }
      
      BigInt((sign << (expBits + mantBits)) | (expEnc << mantBits) | mantEnc)
    }
  }
}

case class UnaryLUTOp[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  valFn: Int => Double,
  encodeFn: Double => BigInt,
  mathFn: Double => Double
) extends Component {
  val bitWidth = dataType.getBitsWidth
  require(bitWidth <= 8, "LUT approach is only for 8-bit or smaller types")
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  // Replicate ROM for each lane to avoid multi-port memory limitations
  val roms = for (i <- 0 until lanes) yield {
    MathLUTs.generateROM(bitWidth, valFn, encodeFn, mathFn)
  }
  
  val outValid = RegInit(False)
  when(io.a.stream.ready) {
    outValid := io.a.stream.valid
  }
  
  val outPayload = Vec(dataType, lanes)
  for (i <- 0 until lanes) {
    val readAddr = io.a.stream.payload(i).asBits.asUInt
    val readData = roms(i).readSync(readAddr, enable = io.a.stream.ready)
    outPayload(i).assignFromBits(readData)
  }
  
  io.a.stream.ready := io.c.stream.ready || !outValid
  io.c.stream.valid := outValid
  io.c.stream.payload := outPayload
}
