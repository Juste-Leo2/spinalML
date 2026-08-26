package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.dtypes.I8
import spinalML.ops.{RepackTestComp, repack}
import spinalML.tensors.Tensor

/**
 * M1 étape B (docs/open-mysteries.md): formal identity proofs for the lane
 * gearbox under FREE handshakes (anyseq valid/ready — no stall-free assume,
 * unlike the original eager RepackFormal spec below).
 *
 * Spec 1 — flushable AGGREGATE 2→4 (I8): a sliding 2-beat window records the
 * accepted input beats; every emission must present exactly wB##wA, and a
 * partial group must never be exposed.
 *
 * Spec 2 — flushable CHAIN 4→1→3 (I8): the non-multiple ratio forces TWO
 * chained gearboxes via lanes=1 (the W4A8 weight-path shape in miniature).
 * Three recorded input beats form a 96-bit flat stream; each of the four
 * 3-byte output groups must equal its static slice of that stream.
 *
 * SPLIT direction stays simulation-covered (RepackStallDiffTest) — noted in
 * open-mysteries M1.
 */

// ---- Spec 2 DUT: non-multiple chain through lanes=1 -----------------------
case class RepackChainComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(12), lanes = 4))
    val c = master(Tensor(I8(), Seq(12), lanes = 3))
  }
  io.c <> repack(io.a, newLanes = 3, withFlush = true)
}

class RepackStallAggregateFormal extends Component {
  val dut = FormalDut(RepackTestComp(I8(), withFlush = true))

  // Free handshakes: stalls are part of the state space now.
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  val inF = dut.io.a.stream.fire
  val outF = dut.io.c.stream.fire
  val inWord = dut.io.a.stream.payload.map(_.asBits).reverse.reduce(_ ## _)
  val outWord = dut.io.c.stream.payload.map(_.asBits).reverse.reduce(_ ## _)

  val pending = RegInit(U(0, 2 bits))
  val wA = Reg(Bits(16 bits))
  val wB = Reg(Bits(16 bits))

  // The two fires are mutually exclusive by construction (ready = !full,
  // valid = full): formalizes the paper argument recorded in
  // docs/open-mysteries.md M1.2-1.
  assert(!(inF && outF), "input and output fired simultaneously")

  // A partial group must never be exposed on the output.
  assert(!(dut.io.c.stream.valid && pending =/= 2))

  // Identity: emission #k carries exactly the k-th and (k+1)-th accepted beats.
  when(outF) {
    assert(pending === 2, "emission with incomplete group")
    assert(outWord === (wB ## wA), "output group mismatch")
  }

  // One emission consumes BOTH buffered beats (m = newLanes / lanes = 2).
  when(inF) { pending := pending + 1 }
  when(outF) { pending := pending - 2 }

  // Fires are mutually exclusive, so the window bookkeeping never overlaps:
  when(inF) {
    when(pending === 0) { wA := inWord } otherwise { wB := inWord }
  } elsewhen (outF) {
    wA := wB
  }

  val emissions = Counter(8, outF)
  cover(emissions.value === 3)        // several groups complete
  cover(dut.io.c.stream.valid && !dut.io.c.stream.ready) // stall on output exercised
}

class RepackChainFormal extends Component {
  val dut = FormalDut(RepackChainComp())

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.c.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  val inF = dut.io.a.stream.fire
  val outF = dut.io.c.stream.fire

  val beatCnt = RegInit(U(0, 2 bits))   // input beats accepted so far (0..3)
  val gIdx = RegInit(U(0, 2 bits))      // output groups emitted so far (0..3)

  val b0 = Reg(Bits(32 bits))
  val b1 = Reg(Bits(32 bits))
  val b2 = Reg(Bits(32 bits))

  // Command-length contract (mirrors FetchRequest): the command carries
  // exactly 3 beats — no further input fire once they are all accepted.
  assume(!(inF && beatCnt === 3))

  when(inF) {
    switch(beatCnt) {
      is(0) { b0 := dut.io.a.stream.payload.map(_.asBits).reverse.reduce(_ ## _) }
      is(1) { b1 := dut.io.a.stream.payload.map(_.asBits).reverse.reduce(_ ## _) }
      is(2) { b2 := dut.io.a.stream.payload.map(_.asBits).reverse.reduce(_ ## _) }
    }
    beatCnt := beatCnt + 1
  }

  // element j sits at byte j of the flat stream; group g = bytes [3g .. 3g+2],
  // i.e. the g-th 24-bit slice LSB-first. NOTE: subdivideIn(n slices) splits
  // into n pieces; subdivideIn(w bits) splits into w-bit pieces — we need the
  // latter (4 slices of 24 bits, not 24 slices of 4 bits!).
  val flat = b2 ## b1 ## b0
  val expectGroup = flat.subdivideIn(24 bits)(gIdx.resized)

  // Group g needs ceil(3*(g+1)/4) accepted beats before it can exist.
  val needLut = Vec(U(1, 2 bits), U(2, 2 bits), U(3, 2 bits), U(3, 2 bits))
  assert(!(dut.io.c.stream.valid && beatCnt < needLut(gIdx)),
    "chain exposes a group whose elements were not all accepted")

  when(outF) {
    assert(dut.io.c.stream.payload.map(_.asBits).reverse.reduce(_ ## _) === expectGroup,
      "chain output group mismatch")
    gIdx := gIdx + 1
  }

  val emissions = Counter(8, outF)
  cover(emissions.value === 3)              // all four groups reachable
  cover(inF && outF)                        // overlap exercised
}

object RepackFormal {
  def main(args: Array[String]): Unit = {
    // M1 étape B: free-handshake identity proofs (supersede the historical
    // eager spec whose assumes pinned valid/ready high).
    FormalConfig
      .withSymbiYosys
      .withBMC(14)
      .withTimeout(1800)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new RepackStallAggregateFormal, "repack_flush_aggregate_stalls")
  }
}

object RepackChainFormalMain {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(16)
      .withTimeout(1800)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new RepackChainFormal, "repack_chain_4to1to3_stalls")
  }
}
