package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, FP4_E2M1, I16, BF16}

case class SoftmaxTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(4, 16), lanes = 4))
  val y = master(Tensor(dataType, Seq(4, 16), lanes = 4))
  
  val comp = Softmax1D(dataType, channels = 4, seqLen = 16)
  comp.io.x <> x
  y <> comp.io.y
}

class SoftmaxTest extends AnyFunSuite {
  test("Softmax1D compilation on I4") { SpinalConfig().generateVerilog(SoftmaxTestComp(I4())) }
  test("Softmax1D compilation on FP4") { SpinalConfig().generateVerilog(SoftmaxTestComp(FP4_E2M1())) }
  test("Softmax1D PWL compilation on I16") { SpinalConfig().generateVerilog(SoftmaxTestComp(I16())) }
  test("Softmax1D PWL compilation on BF16") { SpinalConfig().generateVerilog(SoftmaxTestComp(BF16())) }
}
