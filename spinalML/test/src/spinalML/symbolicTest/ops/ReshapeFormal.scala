package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.ops.ReshapeTestComp
import spinalML.dtypes.I8

class ReshapeFormal extends Component {
  val dut = FormalDut(new ReshapeTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.reshaped.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.reshaped.stream.ready)

  val lanes = dut.io.a.stream.payload.length
  val expected = Vec(I8(), lanes)
  for (lane <- 0 until lanes) {
    expected(lane) := dut.io.a.stream.payload(lane)
  }

  when(dut.io.reshaped.stream.valid && dut.io.reshaped.stream.ready) {
    for (lane <- 0 until lanes) {
      assert(dut.io.reshaped.stream.payload(lane) === expected(lane), s"Reshape lane $lane mismatch")
    }
  }
}

object ReshapeFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new ReshapeFormal, "reshape_i8")
  }
}
