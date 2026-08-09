package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1}

case class LayerNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(4, 16), lanes = 4))
  val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
  val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
  val y = master(Tensor(dataType, Seq(4, 16), lanes = 4))
  
  val comp = LayerNorm1D(dataType, channels = 4, seqLen = 16)
  comp.io.x <> x
  comp.io.gamma <> gamma
  comp.io.beta <> beta
  y <> comp.io.y
}

class LayerNormTest extends AnyFunSuite {
  test("LayerNorm1D skeleton compilation on I4") {
    SpinalConfig().generateVerilog(LayerNormTestComp(I4()))
  }

  test("LayerNorm1D skeleton compilation on FP4") {
    SpinalConfig().generateVerilog(LayerNormTestComp(FP4_E2M1()))
  }
}
