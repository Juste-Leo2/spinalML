package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

class BatchNormTest extends AnyFunSuite {
  test("BatchNorm1D compilation on I8") {
    SpinalConfig().generateVerilog(new Component {
      val x = slave(Tensor(I8(), Seq(4, 16), lanes = 4))
      val gamma = slave(Tensor(I8(), Seq(4), lanes = 4))
      val beta = slave(Tensor(I8(), Seq(4), lanes = 4))
      val y = master(Tensor(I8(), Seq(4, 16), lanes = 4))
      
      y <> batchnorm(x, gamma, beta, seqLen = 16)
    })
  }
}
