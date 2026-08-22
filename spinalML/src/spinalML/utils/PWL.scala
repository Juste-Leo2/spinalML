package spinalML.utils

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

object PWLLUTs {
  // Generates ROM for slopes (a) and intercepts (b)
  // segmentFn(i) should return a tuple of Double (a, b) for the i-th segment
  def generateROMs[T <: Data](numSegments: Int, segmentFn: Int => (Double, Double), dataType: HardType[T]): (Mem[Bits], Mem[Bits]) = {
    val bitWidth = dataType.getBitsWidth
    val isFloat = dataType().isInstanceOf[FloatML]
    
    val encodeFn = if (isFloat) {
      val f = dataType().asInstanceOf[FloatML]
      MathLUTs.floatEncodeFn(f.expBits, f.mantBits)
    } else {
      MathLUTs.intEncodeFn(bitWidth)
    }
    
    val romAContent = for (i <- 0 until numSegments) yield {
      val (a, _) = segmentFn(i)
      B(encodeFn(a), bitWidth bits)
    }
    
    val romBContent = for (i <- 0 until numSegments) yield {
      val (_, b) = segmentFn(i)
      B(encodeFn(b), bitWidth bits)
    }
    
    (Mem(Bits(bitWidth bits), initialContent = romAContent),
     Mem(Bits(bitWidth bits), initialContent = romBContent))
  }

  def createSegmentFn(bitWidth: Int, isFloat: Boolean, expBits: Int, mantBits: Int, indexBits: Int, mathFn: Double => Double): Int => (Double, Double) = {
    val valFn = if (isFloat) MathLUTs.floatValFn(expBits, mantBits) else MathLUTs.intValFn(bitWidth)
    val shift = bitWidth - indexBits

    (i: Int) => {
      val x_start = valFn(i << shift)
      val x_end = valFn(((i + 1) << shift) - 1)

      if (x_start == x_end || x_start.isNaN || x_end.isNaN) {
        val y = mathFn(if(x_start.isNaN) 0.0 else x_start)
        (0.0, if(y.isNaN || y.isInfinity) 0.0 else y)
      } else {
        val y_start = mathFn(x_start)
        val y_end = mathFn(x_end)

        val a = (y_end - y_start) / (x_end - x_start)
        val b = y_start - a * x_start

        val safeA = if (a.isNaN || a.isInfinity) 0.0 else a
        val safeB = if (b.isNaN || b.isInfinity) y_start else b
        (safeA, safeB)
      }
    }
  }

  // Piecewise-CONSTANT fallback for functions whose local slope cannot be
  // represented in the storage dtype (e.g. 1/x near 0 in narrow signed ints:
  // the linear-fit slope saturates at -2^(w-1) while the intercept saturates
  // at +(2^(w-1)-1), yielding garbage like recip(1) = 1*(-2^(w-1)) + (2^(w-1)-1)
  // = -1). Each segment stores (0, f(sample)) where sample is the first
  // abscissa of the segment with |x| >= 1, keeping small positive integers
  // (softmax sums) exact-ish and everything else bounded.
  def createConstantSegmentFn(bitWidth: Int, indexBits: Int, mathFn: Double => Double): Int => (Double, Double) = {
    val maxVal = (1 << (bitWidth - 1)) - 1
    val shift = bitWidth - indexBits

    def signed(raw: Int): Double = {
      val half = 1 << (bitWidth - 1)
      val full = 1 << bitWidth
      if (raw >= half) (raw - full).toDouble else raw.toDouble
    }

    (i: Int) => {
      val xs = signed(i << shift)
      val xe = signed(((i + 1) << shift) - 1)
      // first abscissa of the segment whose magnitude is >= 1 (avoids f's singularity at 0)
      val sample =
        if (xe < 1.0) xe
        else if (xs > 1.0) xs
        else 1.0
      val y = mathFn(sample)
      val ySafe = if (y.isNaN || y.isInfinity || y > maxVal.toDouble || y < -maxVal.toDouble) 0.0 else y
      (0.0, ySafe)
    }
  }
}

case class UnaryPWLOp[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  numSegments: Int,
  segmentIndexFn: T => UInt, // Extracts the ROM index from the input value
  segmentFn: Int => (Double, Double) // Returns (slope a, intercept b) for a segment
) extends Component {
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }
  
  // Replicate ROMs for each lane to allow parallel lookups
  val roms = for (i <- 0 until lanes) yield {
    PWLLUTs.generateROMs(numSegments, segmentFn, dataType)
  }
  
  // Pipeline signals
  val stage1_valid = RegInit(False)
  val stage1_x = Reg(Vec(dataType, lanes))
  
  when(io.a.stream.ready) {
    stage1_valid := io.a.stream.valid
    stage1_x := io.a.stream.payload
  }
  
  val outPayload = Vec(dataType, lanes)
  
  for (i <- 0 until lanes) {
    val x = io.a.stream.payload(i)
    val readAddr = segmentIndexFn(x)
    
    val readA = roms(i)._1.readSync(readAddr, enable = io.a.stream.ready)
    val readB = roms(i)._2.readSync(readAddr, enable = io.a.stream.ready)
    
    val aCoef = dataType()
    aCoef.assignFromBits(readA)
    val bCoef = dataType()
    bCoef.assignFromBits(readB)
    
    // stage1_x contains x aligned with the readSync result
    val xDelayed = stage1_x(i)
    
    // y = a * x + b
    (xDelayed, aCoef) match {
      case (vx: SInt, va: SInt) => 
        val p = (vx * va).resized
        outPayload(i).assignFrom((p + bCoef.asInstanceOf[SInt]).resized.asInstanceOf[T])
      case (vx: UInt, va: UInt) => 
        val p = (vx * va).resized
        outPayload(i).assignFrom((p + bCoef.asInstanceOf[UInt]).resized.asInstanceOf[T])
      case (vx: FloatML, va: FloatML) => 
        val p = Float.mul(vx, va)
        outPayload(i).assignFrom(Float.add(p, bCoef.asInstanceOf[FloatML]).asInstanceOf[T])
      case _ => throw new Exception("Unsupported PWL type")
    }
  }
  
  io.a.stream.ready := io.c.stream.ready || !stage1_valid
  io.c.stream.valid := stage1_valid
  io.c.stream.payload := outPayload
}
