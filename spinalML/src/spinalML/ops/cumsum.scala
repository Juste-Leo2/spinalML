package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

/**
 * CumSumOp: Somme cumulée (Cumulative Sum) le long de la dimension extérieure (L).
 * Idéal pour Mamba2, State Space Models et Linear Attention.
 * Input shape: [L, C] (ex: [SeqLen, Dim])
 */
case class CumSumOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  require(shape.length >= 2, "CumSumOp supporte actuellement les tenseurs d'au moins 2D [..., L, C]")
  
  val rank = shape.length
  val L = shape(rank - 2)
  val C = shape(rank - 1)
  
  val chunks = (C + lanes - 1) / lanes

  val io = new Bundle {
    val in = slave(Tensor(dataType, shape, lanes))
    val out = master(Tensor(dataType, shape, lanes))
  }

  // Compteurs pour suivre la position dans le flux
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

  // La somme cumulée a besoin de l'accumulateur de la ligne précédente (L-1).
  // Puisque les données arrivent en streaming, l'élément (L-1, c) est passé exactement 
  // `chunks` cycles plus tôt ! On utilise un Shift Register (qui se synthétise en SRL très efficace).
  
  val sumResult = Vec(dataType, lanes)
  val prevValDelayed = Delay(sumResult, cycleCount = chunks, when = fire, init = sumResult.getZero)
  
  val prevVal = Mux(isFirstL, sumResult.getZero, prevValDelayed)

  for(i <- 0 until lanes) {
    sumResult(i) := ((io.in.stream.payload(i), prevVal(i)) match {
      case (a: SInt, b: SInt) => (a + b).resized.asInstanceOf[T]
      case (a: UInt, b: UInt) => (a + b).resized.asInstanceOf[T]
      case (a: spinalML.dtypes.FloatML, b: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(a, b).asInstanceOf[T]
      case _ => throw new Exception("Type non supporté pour CumSum")
    })
  }

  // On pipeline la sortie pour absorber le chemin combinatoire de l'additionneur
  io.out.stream << io.in.stream.translateWith(sumResult).m2sPipe()
}

object cumsum {
  def apply[T <: Data](in: Tensor[T]): Tensor[T] = {
    val cumsumComp = CumSumOp(in.dataType, in.shape, in.lanes)
    cumsumComp.io.in <> in
    cumsumComp.io.out
  }
}
