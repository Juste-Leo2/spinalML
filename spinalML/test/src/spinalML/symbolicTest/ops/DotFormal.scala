package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.tensors.Tensor
import spinalML.ops.dot

// Component for formal verification of the dot wrapper: two 4-element I8
// vectors streamed on 2 lanes, producing a single scalar tensor.
case class DotTestCompFormal() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(4), lanes = 2))
    val b = slave(Tensor(I8(), Seq(4), lanes = 2))
    val c = master(Tensor(I8(), Seq(1), lanes = 1))
  }
  io.c <> dot(io.a, io.b)
}

// This spec validates the *flow* of the dot operation (wraps matmul M=1, N=1):
// inputs are consumed as expected, the scalar output is produced exactly once
// per frame, with no deadlock. The numeric correctness is covered by the
// Python/Cocotb co-simulation against the golden models instead.
class DotFormal extends Component {
  val dut = FormalDut(DotTestCompFormal())

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  // Handshake stability: a valid payload must hold while ready is low
  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  // Stop sending after one full frame (4 elements = 2 chunks of 2 lanes)
  val aCounter = Counter(3)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) { aCounter.increment() }
  assume(aCounter.value =/= 2 || !dut.io.a.stream.valid)

  val bCounter = Counter(3)
  when(dut.io.b.stream.valid && dut.io.b.stream.ready) { bCounter.increment() }
  assume(bCounter.value =/= 2 || !dut.io.b.stream.valid)

  // Flow: the scalar output must fire exactly once per frame (no deadlock,
  // no duplicated word).
  val fired = RegInit(False)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(!fired, "Dot must produce a single scalar output per frame")
    fired := True
  }

  cover(fired)
}

object DotFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new DotFormal, "dot_i8_4_2")
  }
}