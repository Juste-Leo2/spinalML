package spinalML.utils

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

object MathLUTs {
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

  // Generates a ROM specifically for FloatML mantissa fraction processing (Algebraic Separation)
  def generateFloatMantissaROM(inBits: Int, outBits: Int, mathFn: Double => Double): Mem[Bits] = {
    val states = 1 << inBits
    val romContent = for (i <- 0 until states) yield {
      val mantFraction = i.toDouble / states
      val realMant = 1.0 + mantFraction
      val y = mathFn(realMant) // y should ideally be within [1.0, 2.0)
      val frac = y - 1.0
      val outStates = 1 << outBits
      val encoded = Math.round(frac * outStates).toInt
      B(if (encoded >= outStates) outStates - 1 else encoded, outBits bits)
    }
    Mem(Bits(outBits bits), initialContent = romContent)
  }

  // Integer codecs (2's complement)
  def intValFn(bitWidth: Int): Int => Double = i => {
    val maxVal = 1 << bitWidth
    val halfVal = 1 << (bitWidth - 1)
    if (i >= halfVal) (i - maxVal).toDouble else i.toDouble
  }
  
  def intEncodeFn(bitWidth: Int): Double => BigInt = y => {
    val maxVal = (1 << (bitWidth - 1)) - 1
    val minVal = -(1 << (bitWidth - 1))
    val clamped = Math.max(minVal.toDouble, Math.min(maxVal.toDouble, Math.round(y)))
    val intVal = clamped.toInt
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
  
  def floatEncodeFn(expBits: Int, mantBits: Int): Double => BigInt = y => {
    if (y == 0.0) BigInt(0)
    else {
      val sign = if (y < 0) 1 else 0
      val absY = Math.abs(y)
      val bias = (1 << (expBits - 1)) - 1
      
      var exp = Math.floor(Math.log(absY) / Math.log(2.0)).toInt
      var mant = (absY / Math.pow(2.0, exp)) - 1.0
      
      var expEnc = exp + bias
      var mantEnc = Math.round(mant * (1 << mantBits)).toInt
      
      if (mantEnc == (1 << mantBits)) { // Rounding overflow
         mantEnc = 0
         expEnc += 1
      }
      
      if (expEnc >= (1 << expBits)) { // Overflow (Sat)
        expEnc = (1 << expBits) - 1
        mantEnc = (1 << mantBits) - 1
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
