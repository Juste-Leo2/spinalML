package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.ops.AddTestComp
import spinalML.dtypes.I8

/**
 * Formal specification of the AddOp (I8) — proves for ALL possible inputs
 * that the streamed result equals the bit-exact wrap-around addition.
 *
 * Run with: ./mill spinalML.test.runMain spinalML.symbolicTest.ops.AddFormal
 */
class AddFormal extends Component {
  val dut = FormalDut(new AddTestComp(I8()))

  // Drive the DUT inputs with free symbolic variables (explore ALL values)
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.payload)

  // Ensure the state space starts with a proper reset
  assumeInitial(clockDomain.isResetActive)

  // --- Domain constraints (legal environment of the op) ---
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  // Keep the inputs stable during the transaction so the pipelined
  // (m2sPipe) output can be compared against the same input pair.
  assume(dut.io.a.stream.payload === past(dut.io.a.stream.payload))
  assume(dut.io.b.stream.payload === past(dut.io.b.stream.payload))

  // --- Spec: bit-exact I8 wrap-around addition (golden model) ---
  val expected0 = (dut.io.a.stream.payload(0) + dut.io.b.stream.payload(0)).resize(8 bits)
  val expected1 = (dut.io.a.stream.payload(1) + dut.io.b.stream.payload(1)).resize(8 bits)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected0, "Add lane 0 mismatch (I8)")
    assert(dut.io.c.stream.payload(1) === expected1, "Add lane 1 mismatch (I8)")
  }
}

object AddFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(2)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AddFormal, "add_i8")
  }
}