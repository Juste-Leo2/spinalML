// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

/**
 * CumSumOp: cumulative sum along the outer (L) dimension.
 * Ideal for Mamba2, State Space Models and Linear Attention.
 * Input shape: [L, C] (e.g. [SeqLen, Dim])
 */
case class CumSumOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  require(shape.length >= 2, "CumSumOp currently supports tensors of at least 2D [..., L, C]")
  
  val rank = shape.length
  val L = shape(rank - 2)
  val C = shape(rank - 1)
  
  val chunks = (C + lanes - 1) / lanes

  val io = new Bundle {
    val in = slave(Tensor(dataType, shape, lanes))
    val out = master(Tensor(dataType, shape, lanes))
  }

  // Position counters tracking the stream location
  val chunkCounter = Counter(chunks)
  val lCounter = Counter(L)
  
  val fire = io.in.stream.fire
  when(fire) {
    chunkCounter.increment()
    when(chunkCounter.willOverflowIfInc) {
      lCounter.increment()
    }
  }
  
  val isFirstL = lCounter.value === 0

  // The cumsum needs the previous-row (L-1) accumulator. Since data
  // streams in, element (L-1, c) passed exactly `chunks` cycles earlier!
  // A shift register is used (synthesizes to efficient SRL).
  
  val sumResult = Vec(dataType, lanes)
  val prevValDelayed = Delay(sumResult, cycleCount = chunks, when = fire, init = sumResult.getZero)
  
  val prevVal = Mux(isFirstL, sumResult.getZero, prevValDelayed)

  for(i <- 0 until lanes) {
    sumResult(i) := ((io.in.stream.payload(i), prevVal(i)) match {
      case (a: SInt, b: SInt) => (a + b).resized.asInstanceOf[T]
      case (a: UInt, b: UInt) => (a + b).resized.asInstanceOf[T]
      case (a: spinalML.dtypes.FloatML, b: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(a, b).asInstanceOf[T]
      case _ => throw new Exception("Unsupported data type for CumSum")
    })
  }

  // Pipeline the output to absorb the adder's combinational path
  io.out.stream << io.in.stream.translateWith(sumResult).m2sPipe()
}

object cumsum {
  def apply[T <: Data](in: Tensor[T]): Tensor[T] = {
    val cumsumComp = CumSumOp(in.dataType, in.shape, in.lanes)
    cumsumComp.io.in <> in
    cumsumComp.io.out
  }
}
