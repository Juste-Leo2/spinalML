package spinalML.activations

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

case class SoftmaxTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(16, 4), lanes = 4))
    val y = master(Tensor(dataType, Seq(16, 4), lanes = 4))
  }
  
  val comp = Softmax1D(dataType, channels = 4, seqLen = 16)
  comp.io.x <> io.x
  io.y <> comp.io.y
}

class SoftmaxTest extends AnyFunSuite {
  test("Softmax1D compilation on I8") { SpinalConfig().generateVerilog(SoftmaxTestComp(I8())) }
  test("Softmax1D compilation on I16") { SpinalConfig().generateVerilog(SoftmaxTestComp(I16())) }
  test("Softmax1D compilation on FP8") { SpinalConfig().generateVerilog(SoftmaxTestComp(FP8_E4M3())) }
  test("Softmax1D compilation on BF16") { SpinalConfig().generateVerilog(SoftmaxTestComp(BF16())) }
}
