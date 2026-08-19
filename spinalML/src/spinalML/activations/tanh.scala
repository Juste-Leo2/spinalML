package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.MulOp

/**
 * TanhOp: tanh(x) = 2 * sigmoid(2x) - 1
 *
 * 1. Stream multiply by 2.0 (MulOp, literal stream)
 * 2. SigmoidOp
 * 3. Combinational ×2 - 1 on the output (same style as Softmax1D's final multiply)
 */
case class TanhOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  val isFloat = dataType().isInstanceOf[FloatML]
  val bitWidth = dataType.getBitsWidth

  // 1. Literal stream of 2.0
  val twoStream = Stream(Vec(dataType, lanes))
  val twoPayload = Vec(dataType, lanes)

  for (i <- 0 until lanes) {
    twoPayload(i) match {
      case (vout: FloatML) =>
        vout.sign := False
        vout.exponent := S(vout.bias + 1, vout.expBits + 1 bits).asUInt.resized // 2.0
        vout.mantissa := 0
      case (vout: SInt) =>
        vout := S(2, bitWidth bits)
      case _ => throw new Exception("Tanh supports SInt and FloatML only")
    }
  }

  twoStream.valid := io.a.stream.valid
  twoStream.payload := twoPayload

  // 2. x2 = 2 * x
  val mulComp = MulOp(dataType, shape, lanes)
  mulComp.io.a.stream << io.a.stream
  mulComp.io.b.stream << twoStream
  val x2Stream = mulComp.io.c.stream

  // 3. Sigmoid(2x)
  val sigComp = SigmoidOp(dataType, shape, lanes)
  sigComp.io.a.stream << x2Stream
  val sigStream = sigComp.io.c.stream

  // 4. y = 2 * sigmoid(2x) - 1 (combinational, like Softmax1D's final multiply)
  val outPayload = Vec(dataType, lanes)

  for (i <- 0 until lanes) {
    (sigStream.payload(i), outPayload(i)) match {
      case (vx: FloatML, vout: FloatML) =>
        val two = FloatML(vx.expBits, vx.mantBits)
        two.sign := False
        two.exponent := S(vx.bias + 1, vx.expBits + 1 bits).asUInt.resized
        two.mantissa := 0
        val mulRes = spinalML.utils.Float.mul(vx, two)

        val negOne = FloatML(vx.expBits, vx.mantBits)
        negOne.sign := True // -1.0
        negOne.exponent := S(vx.bias, vx.expBits + 1 bits).asUInt.resized
        negOne.mantissa := 0
        vout := spinalML.utils.Float.add(mulRes, negOne)

      case (vx: SInt, vout: SInt) =>
        val maxVal = (1 << (bitWidth - 1)) - 1
        val minVal = -(1 << (bitWidth - 1))

        val wide = vx.resize(bitWidth + 2 bits)
        val twice = wide << 1

        val clampedTwice = Mux(twice > S(maxVal, bitWidth + 2 bits), S(maxVal, bitWidth + 2 bits),
                          Mux(twice < S(minVal, bitWidth + 2 bits), S(minVal, bitWidth + 2 bits), twice))

        val minusOne = clampedTwice -^ S(1, bitWidth + 2 bits)

        vout := Mux(minusOne > S(maxVal, bitWidth + 2 bits), S(maxVal, bitWidth + 2 bits),
               Mux(minusOne < S(minVal, bitWidth + 2 bits), S(minVal, bitWidth + 2 bits), minusOne)
               ).resized

      case _ => throw new Exception("Tanh supports SInt and FloatML only")
    }
  }

  io.c.stream.valid := sigStream.valid
  sigStream.ready := io.c.stream.ready
  io.c.stream.payload := outPayload
}

object tanh {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    val comp = TanhOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}