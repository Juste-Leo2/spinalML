package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}

case class LayerNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(16, 4), lanes = 4))
  val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
  val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
  val y = master(Tensor(dataType, Seq(16, 4), lanes = 4))
  
  val comp = LayerNorm1D(dataType, channels = 4, seqLen = 16)
  comp.io.x <> x
  comp.io.gamma <> gamma
  comp.io.beta <> beta
  y <> comp.io.y
}

class LayerNormTest extends AnyFunSuite {
  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"LayerNorm1D compilation on $name") {
      SpinalConfig().generateVerilog(LayerNormTestComp(dt()))
    }
  }
}
