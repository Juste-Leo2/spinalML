// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

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
  val comp = ClassicalAttentionHW(dataType, dataType, accType, seqLen, embedDim, 1, xLanes, wLanes)
  
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
  val comp = ClassicalAttentionHW(dataType, dataType, accType, seqLen, embedDim, numHeads, xLanes, wLanes, projLanes)
  
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

// Weight-only quantization (wXaY): SInt weights + shared compile-time scale(s),
// float activations. accType = activation dtype.
case class AttentionQuantTestComp[T <: Data](
  dataType: HardType[T],
  weightDt: HardType[SInt],
  seqLen: Int,
  embedDim: Int,
  xLanes: Int,
  wLanes: Int,
  scales: Seq[Double] = Seq(1.0)
) extends Component {
  val comp = ClassicalAttentionHW(dataType, weightDt, dataType, seqLen, embedDim, 1, xLanes, wLanes, weightScales = scales)
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(dataType, Seq(seqLen, embedDim), lanes = 1))
  }
  
  comp.io.x <> io.x
  comp.io.wq <> io.wq
  comp.io.wk <> io.wk
  comp.io.wv <> io.wv
  comp.io.wo <> io.wo
  io.y <> comp.io.y
}

case class MultiHeadQuantTestComp[T <: Data](
  dataType: HardType[T],
  weightDt: HardType[SInt],
  seqLen: Int,
  embedDim: Int,
  numHeads: Int,
  xLanes: Int,
  wLanes: Int,
  projLanes: Int,
  scales: Seq[Double] = Seq(1.0)
) extends Component {
  val comp = ClassicalAttentionHW(dataType, weightDt, dataType, seqLen, embedDim, numHeads, xLanes, wLanes, projLanes, scales)
  
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(weightDt, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(dataType, Seq(seqLen, embedDim), lanes = 1))
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

  val quantCombos = Seq(
    ("w8a16", () => I8(), () => BF16()),
    ("w4a16", () => I4(), () => BF16()),
    ("w8a8", () => I8(), () => FP8_E4M3()),
    ("w4a8", () => I4(), () => FP8_E4M3()),
    ("w8a4", () => I8(), () => FP4_E2M1()),
    ("w4a4", () => I4(), () => FP4_E2M1())
  )

  for ((combo, wd, ad) <- quantCombos) {
    test(s"Quant compilation $combo") {
      spinal.core.SpinalVerilog(AttentionQuantTestComp(ad(), wd(), 2, 2, 2, 2))
    }
  }

  test("Quant compilation PerChannel") {
    // embedDim = 2 -> two weight columns -> two scales
    spinal.core.SpinalVerilog(AttentionQuantTestComp(BF16(), I8(), 2, 2, 2, 2, Seq(0.5, 2.0)))
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

  val quantCombos = Seq(
    ("w8a16", () => I8(), () => BF16()),
    ("w4a16", () => I4(), () => BF16()),
    ("w8a8", () => I8(), () => FP8_E4M3()),
    ("w4a8", () => I4(), () => FP8_E4M3()),
    ("w8a4", () => I8(), () => FP4_E2M1()),
    ("w4a4", () => I4(), () => FP4_E2M1())
  )

  for ((combo, wd, ad) <- quantCombos) {
    test(s"Quant compilation $combo") {
      spinal.core.SpinalVerilog(MultiHeadQuantTestComp(ad(), wd(), 4, 4, 2, 4, 4, 4))
    }
  }

  test("Quant compilation PerChannel") {
    // embedDim = 4 -> four weight columns -> four scales
    spinal.core.SpinalVerilog(MultiHeadQuantTestComp(BF16(), I8(), 4, 4, 2, 4, 4, 4, Seq(0.5, -0.25, 1.5, 2.0)))
  }
}
