package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}

case class BatchNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(4, 16), lanes = 4))
  val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
  val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
  val y = master(Tensor(dataType, Seq(4, 16), lanes = 4))
  
  y <> batchnorm(x, gamma, beta, seqLen = 16)
}

class BatchNormTest extends AnyFunSuite {
  test("BatchNorm1D compilation on I4") {
    SpinalConfig().generateVerilog(BatchNormTestComp(I4()))
  }

  test("BatchNorm1D compilation on I16") {
    SpinalConfig().generateVerilog(BatchNormTestComp(I16()))
  }

  test("BatchNorm1D compilation on FP4") {
    SpinalConfig().generateVerilog(BatchNormTestComp(FP4_E2M1()))
  }

  test("BatchNorm1D compilation on BF16") {
    SpinalConfig().generateVerilog(BatchNormTestComp(BF16()))
  }
}
