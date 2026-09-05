// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.utils.{MathLUTs, UnaryLUTOp}

case class ExpOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val bitWidth = dataType.getBitsWidth
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  if (bitWidth <= 8) {
    val isFloat = dataType().isInstanceOf[FloatML]
    val (valFn, encodeFn) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (MathLUTs.floatValFn(f.expBits, f.mantBits), MathLUTs.floatEncodeFn(f.expBits, f.mantBits))
    } else {
      (MathLUTs.intValFn(bitWidth), MathLUTs.intEncodeFn(bitWidth))
    }
    
    val lutOp = UnaryLUTOp(dataType, shape, lanes, valFn, encodeFn, Math.exp)
    lutOp.io.a <> io.a
    io.c <> lutOp.io.c
    
  } else if (dataType().isInstanceOf[FloatML]) {
    // Algebraic separation for FloatML > 8 bits (e.g. BF16, FP16)
    val fType = dataType().asInstanceOf[FloatML]
    val expBits = fType.expBits
    val mantBits = fType.mantBits
    val bias = fType.bias
    
    // The fixed-point math produces a fractional part of 8 bits for the LUT index.
    // This is sufficient for up to BF16 (mantBits=7). For FP16/32, we would need 
    // a larger index (e.g. 10 or 23 bits) and likely PWL interpolation to avoid huge ROMs.
    val lutIndexBits = 8
    
    val mantLuts = for (i <- 0 until lanes) yield {
      spinalML.utils.MathLUTs.generateFloatMantissaROM(lutIndexBits, mantBits, x => Math.pow(2.0, x - 1.0))
    }
    
    val outPayload = Vec(dataType, lanes)
    val stage1_valid = RegInit(False)
    
    when(io.a.stream.ready) {
      stage1_valid := io.a.stream.valid
    }
    
    for (i <- 0 until lanes) {
      val x = io.a.stream.payload(i).asInstanceOf[FloatML]
      val isZero = x.exponent === 0
      
      // 1. Convert to Fixed Point Q8.8
      val expTrueSInt = x.exponent.intoSInt - bias
      val shiftSInt = expTrueSInt - mantBits + 8
      
      val isLeftShift = shiftSInt > 0
      val shiftAbs = Mux(isLeftShift, shiftSInt, -shiftSInt).asUInt
      
      val mantWithOne = (B"1" ## x.mantissa).asUInt
      val absFixed = UInt(16 bits)
      
      when(isLeftShift) {
        absFixed := Mux(shiftAbs > 15, U(0xFFFF, 16 bits), (mantWithOne << shiftAbs.resize(4)).resize(16))
      } otherwise {
        absFixed := Mux(shiftAbs > 15, U(0, 16 bits), (mantWithOne >> shiftAbs.resize(4)).resize(16))
      }
      
      val fixedX = Mux(x.sign, -absFixed.intoSInt, absFixed.intoSInt)
      
      // 2. Multiply by log2(e) in Q0.16 format. log2(e) * 2^16 = 94548
      val log2e = S(94548, 18 bits)
      val yFixedFull = (fixedX * log2e) // Q8.8 * Q0.16 = Q8.24
      
      // 3. Extract Integer (I) and Fractional (F)
      val I = (yFixedFull >> 24).resize(expBits + 2 bits) // SInt
      // We extract the top `lutIndexBits` from the 24-bit fractional part
      val F = yFixedFull(23 downto (24 - lutIndexBits)).asUInt // UInt for LUT index
      
      // 4. LUT Lookup
      val readMant = mantLuts(i).readSync(F, enable = io.a.stream.ready)
      
      val outX = FloatML(expBits, mantBits)
      outX.sign := False // e^x is always positive
      
      val newExpSInt = I + S(bias, expBits + 2 bits)
      val expUnderflow = RegNextWhen(newExpSInt <= 0, io.a.stream.ready)
      val expOverflow = RegNextWhen(newExpSInt >= S((1 << expBits) - 1, expBits + 2 bits), io.a.stream.ready)
      val regNewExp = RegNextWhen(newExpSInt.asUInt.resize(expBits), io.a.stream.ready)
      val regIsZero = RegNextWhen(isZero, io.a.stream.ready)
      
      when(regIsZero) {
        outX.exponent := bias
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
    // PWL Approximation for BF16/FP16/I16/I32
    val indexBits = 8
    val numSegments = 1 << indexBits
    val isFloat = dataType().isInstanceOf[FloatML]
    
    val segmentIndexFn: T => UInt = (x: T) => {
      x.asBits(bitWidth - 1 downto bitWidth - indexBits).asUInt
    }
    
    val (expBits, mantBits) = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      (f.expBits, f.mantBits)
    } else (0, 0)
    
    val mathFn = Math.exp _
    val segmentFn = spinalML.utils.PWLLUTs.createSegmentFn(bitWidth, isFloat, expBits, mantBits, indexBits, mathFn)
    
    val pwlOp = spinalML.utils.UnaryPWLOp(dataType, shape, lanes, numSegments, segmentIndexFn, segmentFn)
    pwlOp.io.a <> io.a
    io.c <> pwlOp.io.c
  }
}

object exp {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = ExpOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
