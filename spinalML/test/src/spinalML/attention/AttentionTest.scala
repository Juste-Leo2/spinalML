package spinalML.layers

import spinalML.attention._
import spinal.core._
import spinal.lib._
import org.scalatest.funsuite.AnyFunSuite
import spinal.core.sim._
import spinalML.tensors.Tensor

case class AttentionTestComp[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  seqLen: Int,
  embedDim: Int,
  xLanes: Int,
  wLanes: Int
) extends Component {
  val comp = ClassicalAttentionHW(dataType, accType, seqLen, embedDim, 1, xLanes, wLanes)
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(accType, Seq(seqLen, embedDim), lanes = 1))
  }
  
  comp.io.x <> io.x
  comp.io.wq <> io.wq
  comp.io.wk <> io.wk
  comp.io.wv <> io.wv
  comp.io.wo <> io.wo
  io.y <> comp.io.y
}

case class MultiHeadAttentionTestComp[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  seqLen: Int,
  embedDim: Int,
  numHeads: Int,
  xLanes: Int,
  wLanes: Int,
  projLanes: Int
) extends Component {
  val comp = ClassicalAttentionHW(dataType, accType, seqLen, embedDim, numHeads, xLanes, wLanes, projLanes)
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(accType, Seq(seqLen, embedDim), lanes = 1))
  }
  
  comp.io.x <> io.x
  comp.io.wq <> io.wq
  comp.io.wk <> io.wk
  comp.io.wv <> io.wv
  comp.io.wo <> io.wo
  io.y <> comp.io.y
}

class ClassicalAttentionTest extends AnyFunSuite {
  import spinalML.dtypes._
  
  val compileTypes = Seq(
    ("I8", () => I8()),
    ("I16", () => I16()),
    ("FP8", () => FP8_E4M3()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    val accDt = if (name == "I8" || name == "I16") () => I32() else dt
    test(name) {
      spinal.core.SpinalVerilog(AttentionTestComp(dt(), accDt(), 2, 2, 2, 2))
    }
  }
}

class MultiHeadAttentionTest extends AnyFunSuite {
  import spinalML.dtypes._
  
  val compileTypes = Seq(
    ("I8", () => I8()),
    ("I16", () => I16()),
    ("FP8", () => FP8_E4M3()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    val accDt = if (name == "I8" || name == "I16") () => I32() else dt
    test(name) {
      spinal.core.SpinalVerilog(MultiHeadAttentionTestComp(dt(), accDt(), 4, 4, 2, 4, 4, 4))
    }
  }
}
