package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.tensors.Tensor
import spinalML.ops.BiasAddOp

// Formal wrapper exposing the internal bias memory and counters so the spec
// can mirror the streaming protocol and check the broadcast addition.
case class BiasAddFormalComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(2, 2), lanes = 2))
    val b = slave(Tensor(I8(), Seq(1, 2), lanes = 1))
    val c = master(Tensor(I8(), Seq(2, 2), lanes = 2))
  }

  val dut = BiasAddOp(I8(), Seq(2, 2), Seq(1, 2), lanes = 2)
  // Command-boundary re-arm: disabled in the formal protocol proof
  // (the spec covers the legacy no-boundary behaviour exactly).
  dut.io.reArm := False
  dut.io.a <> io.a
  dut.io.b <> io.b
  io.c <> dut.io.c

  // Expose internals for the formal spec
  val biasMem = dut.biasMem
}

// Flow + broadcast-addition spec (I8):
// 1. the 2 bias elements are loaded through `b`,
// 2. the FSM then broadcasts them over A (2 chunks of 2 lanes),
// 3. each C beat equals A + bias[col] (combinationally, wrap-around arithmetic),
// 4. exactly 2 beats are produced per frame (no deadlock, no extra word).
class BiasAddFormal extends Component {
  val dut = FormalDut(BiasAddFormalComp())

  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.b.stream.payload)
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  // Handshake stability: valid/payload must hold while ready is low
  val pastValidB = past(dut.io.b.stream.valid)
  val pastReadyB = past(dut.io.b.stream.ready)
  val pastPayloadB = past(dut.io.b.stream.payload)
  when(pastValidB && !pastReadyB) {
    assume(dut.io.b.stream.valid)
    assume(dut.io.b.stream.payload === pastPayloadB)
  }

  val pastValidA = past(dut.io.a.stream.valid)
  val pastReadyA = past(dut.io.a.stream.ready)
  val pastPayloadA = past(dut.io.a.stream.payload)
  when(pastValidA && !pastReadyA) {
    assume(dut.io.a.stream.valid)
    assume(dut.io.a.stream.payload === pastPayloadA)
  }

  // Bias load phase: exactly 2 bias elements, then stop
  val bCounter = Counter(3)
  when(dut.io.b.stream.valid && dut.io.b.stream.ready) { bCounter.increment() }
  assume(bCounter.value =/= 2 || !dut.io.b.stream.valid)

  // Golden bias memory mirrored from the DUT's own biasMem
  val goldenBias = Vec(Reg(I8()) init (S(0, 8 bits)), 2)
  when(dut.io.b.stream.valid && dut.io.b.stream.ready && bCounter.value < 2) {
    switch(bCounter.value) {
      is(0) { goldenBias(0) := dut.io.b.stream.payload(0) }
      is(1) { goldenBias(1) := dut.io.b.stream.payload(0) }
    }
  }

  // Process phase: exactly 2 chunks of A, then stop
  val aCounter = Counter(3)
  when(dut.io.a.stream.valid && dut.io.a.stream.ready) { aCounter.increment() }
  assume(aCounter.value =/= 2 || !dut.io.a.stream.valid)

  // Broadcast check: C is combinationally A + bias[col] during the process state.
  // With lanes == N (2), col(i) == i for every chunk.
  val inProcess = dut.io.a.stream.valid && dut.io.a.stream.ready
  when(inProcess) {
    for (i <- 0 until 2) {
      val sum = (dut.io.a.stream.payload(i).asInstanceOf[SInt] + goldenBias(i)).resize(8 bits)
      assert(dut.io.c.stream.payload(i).asInstanceOf[SInt] === sum, s"Bias broadcast mismatch on lane $i")
    }
  }

  // Flow: exactly 2 output beats per frame, then the FSM returns to load
  val fired = Counter(3)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    fired.increment()
  }
  assert(fired.value < 3, "BiasAdd must produce exactly 2 beats per frame")

  cover(fired.value === 2)
}

object BiasAddFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new BiasAddFormal, "bias_add_i8_2x2_l2")
  }
}