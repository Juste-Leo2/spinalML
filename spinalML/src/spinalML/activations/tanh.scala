// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.MulOp
import spinalML.{RoundingConfig, RoundingMode}

/**
 * TanhOp: tanh(x)
 *
 * FloatML: tanh(x) = 2 * sigmoid(2x) - 1
 *   1. Stream multiply by 2.0 (MulOp, literal stream)
 *   2. SigmoidOp
 *   3. Combinational ×2 - 1 on the output (same style as Softmax1D's final multiply)
 *
 * 8-bit integers (I8/U8): quantized TFLite TANH convention — a full-range LUT
 * over the input codes with `x = (q - inputZeroPoint) * inputScale` and the
 * TFLite output quantization (scale 1/128, zp 0 for int8 / 128 for uint8).
 * See [[QuantActivation]]. 16-bit integers are refused (64K-entry ROM).
 */
case class TanhOp[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  inputScale: Double = 1.0,
  inputZeroPoint: Int = 0,
  rounding: RoundingMode = RoundingConfig.current
) extends Component {

  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, lanes))
  }

  val bitWidth = dataType.getBitsWidth

  dataType() match {
    case _: FloatML => floatDatapath()
    case _: SInt =>
      require(bitWidth == 8,
        s"TanhOp integer path is defined for 8-bit codes only (TFLite int8/uint8 conventions), got ${bitWidth}-bit. " +
        "Use FloatML for wider activations.")
      quantizedDatapath()
    case _: UInt =>
      require(bitWidth == 8,
        s"TanhOp integer path is defined for 8-bit codes only (TFLite int8/uint8 conventions), got ${bitWidth}-bit. " +
        "Use FloatML for wider activations.")
      quantizedDatapath()
    case _ => throw new Exception("Tanh supports SInt, UInt and FloatML only")
  }

  private def quantizedDatapath(): Unit = {
    val signed = dataType().isInstanceOf[SInt]
    val lut = QuantActivation.lut(
      dataType, shape, lanes,
      inputScale, inputZeroPoint,
      QuantActivation.tanhOutputScale, QuantActivation.tanhOutputZeroPoint(signed),
      rounding
    )(x => Math.tanh(x))
    lut.io.a <> io.a
    io.c <> lut.io.c
  }

  private def floatDatapath(): Unit = {
    val twoStream = Stream(Vec(dataType, lanes))
    val twoPayload = Vec(dataType, lanes)

    for (i <- 0 until lanes) {
      twoPayload(i) match {
        case (vout: FloatML) =>
          vout.sign := False
          vout.exponent := S(vout.bias + 1, vout.expBits + 1 bits).asUInt.resized // 2.0
          vout.mantissa := 0
        case _ => throw new Exception("Tanh float path supports FloatML only")
      }
    }

    twoStream.valid := io.a.stream.valid
    twoStream.payload := twoPayload

    // x2 = 2 * x
    val mulComp = MulOp(dataType, shape, lanes)
    mulComp.io.a.stream << io.a.stream
    mulComp.io.b.stream << twoStream
    val x2Stream = mulComp.io.c.stream

    val sigComp = SigmoidOp(dataType, shape, lanes)
    sigComp.io.a.stream << x2Stream
    val sigStream = sigComp.io.c.stream

    // y = 2 * sigmoid(2x) - 1 (combinational, like Softmax1D's final multiply)
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

        case _ => throw new Exception("Tanh float path supports FloatML only")
      }
    }

    io.c.stream.valid := sigStream.valid
    sigStream.ready := io.c.stream.ready
    io.c.stream.payload := outPayload
  }
}

object tanh {
  def apply[T <: Data](
    a: Tensor[T],
    inputScale: Double = 1.0,
    inputZeroPoint: Int = 0,
    rounding: RoundingMode = RoundingConfig.current
  ): Tensor[T] = {
    val comp = TanhOp(a.dataType, a.shape, a.lanes, inputScale, inputZeroPoint, rounding)
    comp.io.a <> a
    comp.io.c
  }
}
