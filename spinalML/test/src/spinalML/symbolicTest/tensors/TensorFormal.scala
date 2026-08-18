package spinalML.symbolicTest.tensors

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

case class TensorTestComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor.Tensor1D(I8(), ne0 = 10, lanes = 2))
    val b = master(Tensor.Tensor1D(I8(), ne0 = 10, lanes = 2))
  }
  
  // Connect stream A to stream B directly
  io.b.stream << io.a.stream
}

class TensorFormal extends Component {
  val dut = FormalDut(TensorTestComp())

  // Drive inputs
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.b.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)

  // Domain constraints
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.ready)
  
  // Golden model / Assertions
  // Since they are directly connected, the output stream must exactly match the input stream
  when(dut.io.b.stream.valid) {
    assert(dut.io.b.stream.payload(0) === dut.io.a.stream.payload(0), "Tensor stream lane 0 mismatch")
    assert(dut.io.b.stream.payload(1) === dut.io.a.stream.payload(1), "Tensor stream lane 1 mismatch")
  }
  
  // Assert valid/ready propagation
  assert(dut.io.b.stream.valid === dut.io.a.stream.valid, "Tensor valid propagation mismatch")
  assert(dut.io.a.stream.ready === dut.io.b.stream.ready, "Tensor ready propagation mismatch")
}

object TensorFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(2)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new TensorFormal, "tensor_formal")
  }
}
