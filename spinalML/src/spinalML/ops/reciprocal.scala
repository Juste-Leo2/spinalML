package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class ReciprocalOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val bitWidth = dataType.getBitsWidth
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  // mathFn with divide-by-zero protection
  val mathFn = (x: Double) => 1.0 / (x + (if (x >= 0) 1e-9 else -1e-9))

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
    val fType = dataType().asInstanceOf[FloatML]
    val expBits = fType.expBits
    val mantBits = fType.mantBits
    val bias = fType.bias
    
    // LUT for mantissa: maps 1.M to 2 / 1.M
    // The mantissa has `mantBits`. M ranges from 0 to (1<<mantBits)-1.
    val numEntries = 1 << mantBits
    val lutContent = for (m <- 0 until numEntries) yield {
      if (m == 0) {
        B(0, mantBits bits)
      } else {
        val floatM = 1.0 + m.toDouble / numEntries
        val recipM = 2.0 / floatM // This is in (1.0, 2.0)
        val newM = scala.math.round((recipM - 1.0) * numEntries).toInt
        val clampedM = scala.math.min(scala.math.max(newM, 0), numEntries - 1)
        B(clampedM, mantBits bits)
      }
    }
    
    val mantLuts = for (i <- 0 until lanes) yield Mem(Bits(mantBits bits), initialContent = lutContent)
    
    val outPayload = Vec(dataType, lanes)
    val stage1_valid = RegInit(False)
    
    when(io.a.stream.ready) {
      stage1_valid := io.a.stream.valid
    }
    
    for (i <- 0 until lanes) {
      val x = io.a.stream.payload(i).asInstanceOf[FloatML]
      val isZero = x.exponent === 0
      val mantIsZero = x.mantissa === 0
      
      val readMant = mantLuts(i).readSync(x.mantissa, enable = io.a.stream.ready)
      
      val outX = FloatML(expBits, mantBits)
      outX.sign := RegNextWhen(x.sign, io.a.stream.ready)
      
      val expSInt = x.exponent.intoSInt
      val shift = Mux(mantIsZero, S(0, expBits+2 bits), S(1, expBits+2 bits))
      val newExpSInt = S(2 * bias, expBits+2 bits) - expSInt - shift
      val newExp = newExpSInt.asUInt.resized
      
      val expIsZero = RegNextWhen(isZero, io.a.stream.ready)
      val expUnderflow = RegNextWhen(newExpSInt <= 0, io.a.stream.ready)
      val expOverflow = RegNextWhen(newExpSInt >= ((1 << expBits) - 1), io.a.stream.ready)
      val regNewExp = RegNextWhen(newExp.resize(expBits), io.a.stream.ready)
      
      when(expIsZero) {
        outX.exponent := ((1 << expBits) - 1) // 1/0 = Inf
        outX.mantissa := 0
      } elsewhen (expUnderflow) {
        outX.exponent := 0
        outX.mantissa := 0
      } elsewhen (expOverflow) {
        outX.exponent := ((1 << expBits) - 1)
        outX.mantissa := 0
      } otherwise {
        outX.exponent := regNewExp
        outX.mantissa := readMant.asUInt
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

object reciprocal {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = ReciprocalOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
