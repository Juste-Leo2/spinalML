package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

class LayerNormTest extends AnyFunSuite {
  test("LayerNorm1D skeleton compilation on I8") {
    SpinalConfig().generateVerilog(new Component {
      val x = slave(Tensor(I8(), Seq(4, 16), lanes = 4))
      val gamma = slave(Tensor(I8(), Seq(4), lanes = 4))
      val beta = slave(Tensor(I8(), Seq(4), lanes = 4))
      val y = master(Tensor(I8(), Seq(4, 16), lanes = 4))
      
      val comp = LayerNorm1D(I8(), channels = 4, seqLen = 16)
      comp.io.x <> x
      comp.io.gamma <> gamma
      comp.io.beta <> beta
      y <> comp.io.y
    })
  }
}
