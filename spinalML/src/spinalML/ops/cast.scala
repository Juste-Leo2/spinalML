// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML

/**
 * CastOp: SInt -> FloatML conversion (with optional weight dequantization),
 * or SInt -> SInt sign-extending width conversion.
 *
 * `scales` (compile-time constants) implements the weight-only quantization
 * (wXaY) policy and integer-to-float domain boundaries:
 * W_float = FloatML(W_int) * scale.
 *   - scales.length == 1            : per-tensor scale
 *   - scales.length == totalBeats   : per-channel scale, indexed by stream
 *                                     beat order (for column-major weight
 *                                     streams, beat n == output channel n).
 * Default Seq(1.0) generates the exact same hardware as a pure cast.
 */
case class CastOp[TIn <: Data, TOut <: Data](
  dataTypeIn: HardType[TIn],
  dataTypeOut: HardType[TOut],
  shape: Seq[Int],
  lanes: Int,
  scales: Seq[Double] = Seq(1.0)
) extends Component {

  val io = new Bundle {
    val a = slave(Tensor(dataTypeIn, shape, lanes))
    val c = master(Tensor(dataTypeOut, shape, lanes))
  }

  // Pass through the stream control signals
  io.c.stream.arbitrationFrom(io.a.stream)

  val totalBeats = (shape.product + lanes - 1) / lanes
  require(
    scales.length == 1 || scales.length == totalBeats,
    s"scales must have length 1 (per-tensor) or $totalBeats (per-channel), got ${scales.length}"
  )

  val useScale = !(scales.length == 1 && scales(0) == 1.0)

  // Beat counter to select the per-channel scale (stream beat order)
  val beatCounter = if (useScale && scales.length > 1) Some(Counter(totalBeats)) else None

  // Elaboration-time scale constants (selected once, shared by all lanes).
  // Only built when a scale is actually requested; SInt -> SInt casts skip it.
  val scaleHw: Option[FloatML] = if (!useScale) None else Some {
    val (expBitsOut, mantBitsOut) = dataTypeOut() match {
      case f: FloatML => (f.expBits, f.mantBits)
      case _ => throw new Exception("CastOp with scales requires a FloatML output type")
    }
    val scaleLits = scales.map(s => spinalML.utils.Float.fromDouble(s, expBitsOut, mantBitsOut))
    beatCounter match {
      case Some(cnt) =>
        var acc: FloatML = scaleLits.head
        scaleLits.zipWithIndex.tail.foreach { case (lit, idx) =>
          acc = Mux(cnt.value === U(idx), lit, acc)
        }
        acc
      case None => scaleLits.head
    }
  }

  for (i <- 0 until lanes) {
    (io.a.stream.payload(i), io.c.stream.payload(i)) match {
      case (valIn: SInt, valOut: FloatML) =>
        val converted = spinalML.utils.Float.fromSInt(valIn, valOut.expBits, valOut.mantBits)
        val result = if (useScale) {
          spinalML.utils.Float.mul(converted, scaleHw.get)
        } else {
          converted
        }
        io.c.stream.payload(i).assignFrom(result.asInstanceOf[TOut])
      case (valIn: SInt, valOut: SInt) =>
        require(!useScale, "CastOp SInt -> SInt does not support scales")
        io.c.stream.payload(i).assignFrom(valIn.resize(valOut.getWidth).asInstanceOf[TOut])
      // More cases can be added here if needed in the future (e.g. UInt -> Float, Float -> SInt, etc.)
      case _ =>
        throw new Exception("Type de cast non supporté (SInt -> FloatML et SInt -> SInt sont gérés)")
    }
  }

  beatCounter.foreach { cnt =>
    when(io.c.stream.valid && io.c.stream.ready) {
      cnt.increment()
    }
  }
}

object cast {
  def apply[TIn <: Data, TOut <: Data](a: Tensor[TIn], dataTypeOut: HardType[TOut], scales: Seq[Double] = Seq(1.0)): Tensor[TOut] = {
    val castComp = CastOp(a.dataType, dataTypeOut, a.shape, a.lanes, scales)
    castComp.io.a <> a
    castComp.io.c
  }
}
