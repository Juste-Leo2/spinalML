package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{ExpOp, ReciprocalOp}

/**
 * SigmoidOp: sigmoid(x) = 1 / (1 + e^(-x))
 *
 * Composed from validated primitives (same philosophy as Softmax1D):
 *   sign-negation -> ExpOp -> +1 -> ReciprocalOp
 */
case class SigmoidOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  val isFloat = dataType().isInstanceOf[FloatML]
  val bitWidth = dataType.getBitsWidth

  // 1. Negation: flip the sign bit (FloatML) or 0 - x (SInt), clamped for ints
  val negStream = Stream(Vec(dataType, lanes))
  val negPayload = Vec(dataType, lanes)

  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), negPayload(i)) match {
      case (vx: FloatML, vout: FloatML) =>
        vout.sign := !vx.sign
        vout.exponent := vx.exponent
        vout.mantissa := vx.mantissa
      case (vx: SInt, vout: SInt) =>
        val minVal = -(1 << (bitWidth - 1))
        val neg = S(0, bitWidth + 1 bits) -^ vx
        val clamped = Mux(neg < S(minVal, bitWidth + 1 bits), S(minVal, bitWidth + 1 bits), neg)
        vout := clamped.resized
      case _ => throw new Exception("Sigmoid supports SInt and FloatML only")
    }
  }

  negStream.valid := io.a.stream.valid
  negStream.payload := negPayload
  io.a.stream.ready := negStream.ready

  // 2. Exp: e^(-x)
  val expComp = ExpOp(dataType, shape, lanes)
  expComp.io.a.stream << negStream
  val expOutStream = expComp.io.c.stream
  val expOutVec = expOutStream.payload

  // 3. +1
  val addOneStream = Stream(Vec(dataType, lanes))
  val addOnePayload = Vec(dataType, lanes)

  for (i <- 0 until lanes) {
    (expOutVec(i), addOnePayload(i)) match {
      case (vx: FloatML, vout: FloatML) =>
        val one = FloatML(vx.expBits, vx.mantBits)
        one.sign := False
        one.exponent := S(vx.bias, vx.expBits bits).asUInt.resized
        one.mantissa := 0
        vout := spinalML.utils.Float.add(vx, one)
      case (vx: SInt, vout: SInt) =>
        val maxVal = (1 << (bitWidth - 1)) - 1
        val sum = S(1, bitWidth + 1 bits) +^ vx
        vout := Mux(sum > S(maxVal, bitWidth + 1 bits), S(maxVal, bitWidth + 1 bits), sum).resized
      case _ => throw new Exception("Sigmoid supports SInt and FloatML only")
    }
  }

  addOneStream.valid := expOutStream.valid
  addOneStream.payload := addOnePayload
  expOutStream.ready := addOneStream.ready

  // 4. Reciprocal: 1 / (1 + e^(-x))
  val recipComp = ReciprocalOp(dataType, shape, lanes)
  recipComp.io.a.stream << addOneStream
  io.c.stream << recipComp.io.c.stream
}

object sigmoid {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = SigmoidOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}