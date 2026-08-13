package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class SqrtOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val bitWidth = dataType.getBitsWidth
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  val mathFn = (x: Double) => Math.sqrt(Math.abs(x))

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
    val fType = dataType().asInstanceOf[FloatML]
    val expBits = fType.expBits
    val mantBits = fType.mantBits
    val bias = fType.bias
    
    val lutIndexBits = mantBits + 1
    val lutStates = 1 << lutIndexBits
    
    val sqrtLuts = for (i <- 0 until lanes) yield {
      val romContent = for (idx <- 0 until lutStates) yield {
        val p = (idx >> mantBits) & 1
        val m_int = idx & ((1 << mantBits) - 1)
        val m_frac = m_int.toDouble / (1 << mantBits)
        val x = (1.0 + m_frac) * (if (p == 1) 2.0 else 1.0)
        
        val y = Math.sqrt(x) // y is in [1.0, 2.0)
        
        var m_out_frac = y - 1.0
        var m_out_int = Math.round(m_out_frac * (1 << mantBits)).toInt
        var e_adj = 0
        if (m_out_int >= (1 << mantBits)) {
          m_out_int = 0
          e_adj = 1
        }
        
        B((e_adj << mantBits) | m_out_int, (mantBits + 1) bits)
      }
      Mem(Bits((mantBits + 1) bits), initialContent = romContent)
    }
    
    val outPayload = Vec(dataType, lanes)
    val stage1_valid = RegInit(False)
    
    when(io.a.stream.ready) {
      stage1_valid := io.a.stream.valid
    }
    
    for (i <- 0 until lanes) {
      val x = io.a.stream.payload(i).asInstanceOf[FloatML]
      val isZero = x.exponent === 0
      
      val biasSInt = S(bias, expBits + 2 bits)
      val expSInt = x.exponent.intoSInt.resize(expBits + 2) - biasSInt
      val parity = expSInt.lsb.asUInt
      val lutIndex = (parity @@ x.mantissa)
      
      val readVal = sqrtLuts(i).readSync(lutIndex, enable = io.a.stream.ready)
      
      val outX = FloatML(expBits, mantBits)
      outX.sign := RegNextWhen(x.sign, io.a.stream.ready)
      
      val newExpSInt = (expSInt >> 1)
      val e_adj_bit = readVal(mantBits)
      val e_adj = Mux(e_adj_bit, S(1, expBits+2 bits), S(0, expBits+2 bits))
      
      val finalExpSInt = newExpSInt + e_adj + biasSInt
      
      val regFinalExp = RegNextWhen(finalExpSInt, io.a.stream.ready)
      
      val expUnderflow = RegNextWhen(finalExpSInt <= 0, io.a.stream.ready)
      val expOverflow = RegNextWhen(finalExpSInt >= ((1 << expBits) - 1), io.a.stream.ready)
      val expIsZero = RegNextWhen(isZero, io.a.stream.ready)
      
      when(expIsZero) {
        outX.exponent := 0
        outX.mantissa := 0
      } elsewhen (expUnderflow) {
        outX.exponent := 0
        outX.mantissa := 0
      } elsewhen (expOverflow) {
        outX.exponent := ((1 << expBits) - 1)
        outX.mantissa := 0
      } otherwise {
        outX.exponent := regFinalExp.asUInt.resize(expBits)
        outX.mantissa := readVal(mantBits - 1 downto 0).asUInt
      }
      
      outPayload(i).assignFrom(outX.asInstanceOf[T])
    }
    
    io.a.stream.ready := io.c.stream.ready || !stage1_valid
    io.c.stream.valid := stage1_valid
    io.c.stream.payload := outPayload
    
  } else {
    // PWL Approximation
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

object sqrt {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = SqrtOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
