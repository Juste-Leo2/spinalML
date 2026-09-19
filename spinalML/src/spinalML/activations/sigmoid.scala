// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.activations

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops.{ExpOp, ReciprocalOp}
import spinalML.{RoundingConfig, RoundingMode}

/**
 * SigmoidOp: sigmoid(x) = 1 / (1 + e^(-x))
 *
 * FloatML: composed from validated primitives (same philosophy as Softmax1D):
 *   sign-negation -> ExpOp -> +1 -> ReciprocalOp
 *
 * 8-bit integers (I8/U8): quantized TFLite LOGISTIC convention — a full-range
 * LUT over the input codes with `x = (q - inputZeroPoint) * inputScale` and
 * the TFLite output quantization (scale 1/256, zp -128 for int8 / 0 for
 * uint8). See [[QuantActivation]]. 16-bit integers are refused: the int8
 * conventions do not define them and a 64K-entry ROM is not a LUT anymore.
 */
case class SigmoidOp[T <: Data](
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
        s"SigmoidOp integer path is defined for 8-bit codes only (TFLite int8/uint8 conventions), got ${bitWidth}-bit. " +
        "Use FloatML for wider activations.")
      quantizedDatapath()
    case _: UInt =>
      require(bitWidth == 8,
        s"SigmoidOp integer path is defined for 8-bit codes only (TFLite int8/uint8 conventions), got ${bitWidth}-bit. " +
        "Use FloatML for wider activations.")
      quantizedDatapath()
    case _ => throw new Exception("Sigmoid supports SInt, UInt and FloatML only")
  }

  private def quantizedDatapath(): Unit = {
    val signed = dataType().isInstanceOf[SInt]
    val lut = QuantActivation.lut(
      dataType, shape, lanes,
      inputScale, inputZeroPoint,
      QuantActivation.logisticOutputScale, QuantActivation.logisticOutputZeroPoint(signed),
      rounding
    )(x => 1.0 / (1.0 + Math.exp(-x)))
    lut.io.a <> io.a
    io.c <> lut.io.c
  }

  private def floatDatapath(): Unit = {
    // 1. Negation: flip the sign bit (FloatML)
    val negStream = Stream(Vec(dataType, lanes))
    val negPayload = Vec(dataType, lanes)

    for (i <- 0 until lanes) {
      (io.a.stream.payload(i), negPayload(i)) match {
        case (vx: FloatML, vout: FloatML) =>
          vout.sign := !vx.sign
          vout.exponent := vx.exponent
          vout.mantissa := vx.mantissa
        case _ => throw new Exception("Sigmoid float path supports FloatML only")
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
        case _ => throw new Exception("Sigmoid float path supports FloatML only")
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
}

object sigmoid {
  def apply[T <: Data](
    a: Tensor[T],
    inputScale: Double = 1.0,
    inputZeroPoint: Int = 0,
    rounding: RoundingMode = RoundingConfig.current
  ): Tensor[T] = {
    val comp = SigmoidOp(a.dataType, a.shape, a.lanes, inputScale, inputZeroPoint, rounding)
    comp.io.a <> a
    comp.io.c
  }
}
