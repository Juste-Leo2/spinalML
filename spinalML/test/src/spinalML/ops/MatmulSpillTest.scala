// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{BF16, I8}
import spinalML.replica.HWArithmetic._
import spinalML.replica.LayerReplicas
import org.scalatest.funsuite.AnyFunSuite

// Compute-side spill: the MatmulOp slice engine runs one K-slice per pass
// (each pass looks like a self-contained GEMM over the slice geometry,
// chained by the spill streams). The pass controller is modeled here by the
// testbench: levels + reArm pulse + slice re-fire per pass, spilled partials
// looped back in software (exact DDR model).
// Geometry shared by all spill tests: full K=8 split in P=2 slices of Ks=4,
// M=2, N=3, lanes=2 (slice multiple of lanes).
//
// Bench discipline — DETERMINISTIC settle-then-transfer driving (the only
// rule that survived contact with the Verilator backend: a poke becomes
// transfer-visible one edge AFTER it is set, so poking and transferring on
// the same edge eats stale data — this exact file once computed row 0 from
// a stale payload because of it):
// - tick() = poll passDone/y + exactly one waitSampling, used for EVERY
//   cycle step so no pulse window is ever unpolled.
// - DRIVING: wait-once for the stream window (bounded poll, valid LOW), then
//   per beat: valid=False + payload + tick (settle: the poke lands, and no
//   transfer is possible with valid low), then valid=True + tick (transfers
//   exactly this beat). Duplication is structurally impossible (at most one
//   transfer edge per beat with valid high); a skip fails loud (collect
//   timeout). Windows persist by construction (the DUT cannot leave
//   LoadA/SeedRow/LoadBias without consuming the beats it waits for).
// - COLLECTING (DUT-held valid, testbench-held ready): check-then-advance —
//   sample valid/payload BEFORE tick(), so the last beat (which drops valid
//   on its transfer edge) is still observed.
// - MATMUL ROW LOOP per pass: B slice, seed row 0, A row 0, drain row 0,
//   seed row 1, A row 1, drain row 1 (seed and drain are row-interleaved,
//   mirroring the DUT's windowed SeedRow/EmitRow).
case class MatmulSpillComp[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  M: Int,
  Ks: Int,
  N: Int,
  lanes: Int
) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(M, Ks), lanes))
    val b = slave(Tensor(dataType, Seq(Ks, N), lanes))
    val c = master(Tensor(accType, Seq(M, N), 1))
    val spillIn = slave(Tensor(accType, Seq(M, N), 1))
    val spillOut = master(Tensor(accType, Seq(M, N), 1))
    val passFirst = in Bool()
    val passLast = in Bool()
    val passDone = out Bool()
    val reArm = in Bool()
  }
  io.c <> matmul(io.a, io.b, accType, parallelN = false, reArm = Some(io.reArm), temporal = 1,
    spill = true, passFirst = Some(io.passFirst), passLast = Some(io.passLast),
    spillSource = Some(io.spillIn), spillSink = Some(io.spillOut), passDone = Some(io.passDone))
}

// Bias-zero gate vehicle: BiasAddOp fed with an all-zero bias must be an
// exact passthrough (per dtype). On failure the fallback is a BiasAddOp
// bypass on non-final passes (see LinearLayer).
// NOTE: BiasAddOp is a combinational passthrough (no output parking), so its
// last beat is unobservable with sampling alone. The test-only StreamFifo
// parks every beat (production RTL untouched) and the bench collects with
// check-then-advance: exact observation of all beats incl. the last.
case class BiasZeroComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 4), 2))
    val b = slave(Tensor(dataType, Seq(1, 4), 1))
    val c = master(Tensor(dataType, Seq(2, 4), 2))
    val reArm = in Bool()
  }
  val biasOut = bias_add(io.a, io.b, reArm = Some(io.reArm))
  val fifo = StreamFifo(Vec(dataType, 2), 16)
  fifo.io.push << biasOut.stream
  io.c.stream << fifo.io.pop
}

// LinearLayer spill smoke: slice engine + bias-zero mux + passDone
// threading, I8.
case class LinearSpillComp() extends Component {
  val io = new Bundle {
    val a = slave(Tensor(I8(), Seq(2, 4), 2))
    val w = slave(Tensor(I8(), Seq(4, 3), 2))
    val b = slave(Tensor(I8(), Seq(1, 3), 1))
    val y = master(Tensor(I8(), Seq(2, 3), 1))
    val spillIn = slave(Tensor(I8(), Seq(2, 3), 1))
    val spillOut = master(Tensor(I8(), Seq(2, 3), 1))
    val passFirst = in Bool()
    val passLast = in Bool()
    val passDone = out Bool()
    val reArm = in Bool()
    val biasReArm = in Bool()
  }
  val comp = spinalML.layers.LinearLayer(I8(), I8(), I8(), Seq(2, 4), Seq(4, 3), 2,
    Seq(1.0), 1024, false, 1, true)
  comp.io.reArm := io.reArm
  comp.io.biasReArm := io.biasReArm
  comp.io.passFirst.get := io.passFirst
  comp.io.passLast.get := io.passLast
  io.passDone := comp.io.passDone.get
  comp.io.a.stream << io.a.stream
  comp.io.w.stream << io.w.stream
  comp.io.b.stream << io.b.stream
  io.y.stream << comp.io.y.stream
  comp.io.spillIn.get.stream << io.spillIn.stream
  io.spillOut.stream << comp.io.spillOut.get.stream
}

class MatmulSpillTest extends AnyFunSuite {

  // Full problem: A[2,8] x W[8,3], P=2 slices of Ks=4. Tiny values (no wrap).
  val aFull = Seq(Seq(1, 2, 1, 0, 2, 1, 0, 1), Seq(0, 1, 2, 1, 0, 2, 1, 0))
  val wFull = (0 until 8).map(k => (0 until 3).map(n => ((k + n) % 3) - 1))
  val M = 2
  val N = 3
  val Ks = 4

  // B slice beats, column-major, 2 beats per column.
  def bBeats(pass: Int): Seq[Seq[Int]] = {
    val k0 = pass * Ks
    (0 until N).flatMap(n => Seq(
      Seq(wFull(k0)(n), wFull(k0 + 1)(n)),
      Seq(wFull(k0 + 2)(n), wFull(k0 + 3)(n))))
  }

  // One A row slice: 2 beats.
  def aRowBeats(pass: Int, m: Int): Seq[Seq[Int]] = {
    val k0 = pass * Ks
    Seq(Seq(aFull(m)(k0), aFull(m)(k0 + 1)), Seq(aFull(m)(k0 + 2), aFull(m)(k0 + 3)))
  }

  // Expected partials of one pass, row-major over [M, N].
  def partialRef(pass: Int): Seq[Int] = {
    val k0 = pass * Ks
    (for (m <- 0 until M; n <- 0 until N)
      yield (0 until Ks).map(j => aFull(m)(k0 + j) * wFull(k0 + j)(n)).sum).toSeq
  }

  val fullRef: Seq[Int] =
    (for (m <- 0 until M; n <- 0 until N)
      yield (0 until 8).map(k => aFull(m)(k) * wFull(k)(n)).sum).toSeq

  test("S1 spill plumbing on I8: partials then full GEMM, passDone per pass") {
    SimConfig.withWave.compile(MatmulSpillComp(I8(), I8(), M, Ks, N, 2)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      var doneEdges = 0
      var prevDone = false
      def tick(): Unit = {
        val cur = dut.io.passDone.toBoolean
        if (cur && !prevDone) doneEdges += 1
        prevDone = cur
        dut.clockDomain.waitSampling()
      }
      def settle(cycles: Int): Unit = for (_ <- 0 until cycles) tick()

      // Wait-once for a stream window (bounded); the window then persists
      // until its beats are consumed.
      def waitReadyOnce(getReady: => Boolean, name: String): Unit = {
        var cycles = 0
        while (!getReady && cycles < 800) { tick(); cycles += 1 }
        assert(getReady, s"$name window never opened")
      }

      // Idle levels. Drain readys are armed per pass below (cross-traps: the
      // wrong drain target stalls the pass instead of aliasing data).
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.spillIn.stream.valid #= false
      dut.io.c.stream.ready #= false
      dut.io.spillOut.stream.ready #= false
      dut.io.passFirst #= false
      dut.io.passLast #= false
      dut.io.reArm #= false
      settle(2)

      def firePass(first: Boolean, last: Boolean): Unit = {
        dut.io.passFirst #= first
        dut.io.passLast #= last
        dut.io.reArm #= true
        tick()
        dut.io.reArm #= false
      }

      // Push beats, v3: window waited (valid LOW), beat 0 pre-poked +
      // settle tick, then valid HIGH held and exactly one tick per beat.
      def pushN(beats: Seq[Seq[Int]], setValid: Boolean => Unit, setPayload: Seq[Int] => Unit,
        getReady: => Boolean, name: String): Unit = {
        require(beats.nonEmpty, s"$name: empty push")
        waitReadyOnce(getReady, name)
        setPayload(beats.head)
        tick()
        setValid(true)
        for (i <- beats.indices) {
          tick()
          if (i + 1 < beats.length) setPayload(beats(i + 1))
        }
        setValid(false)
        tick()
      }

      def pushB(beats: Seq[Seq[Int]]): Unit = {
        pushN(beats, v => dut.io.b.stream.valid #= v, beat => {
          dut.io.b.stream.payload(0) #= beat(0)
          dut.io.b.stream.payload(1) #= beat(1)
        }, dut.io.b.stream.ready.toBoolean, "B")
      }

      def pushARow(beats: Seq[Seq[Int]]): Unit = {
        pushN(beats, v => dut.io.a.stream.valid #= v, beat => {
          dut.io.a.stream.payload(0) #= beat(0)
          dut.io.a.stream.payload(1) #= beat(1)
        }, dut.io.a.stream.ready.toBoolean, "A row")
      }

      def pushSpillRow(vals: Seq[Int]): Unit = {
        pushN(vals.map(v => Seq(v)), v => dut.io.spillIn.stream.valid #= v, beat => {
          dut.io.spillIn.stream.payload(0) #= beat(0)
        }, dut.io.spillIn.stream.ready.toBoolean, "spill seed")
      }

      // Collect exactly n beats check-then-advance from the live sink.
      def collectN(n: Int, getValid: => Boolean, getPayload: => Int, name: String): Seq[Int] = {
        val got = scala.collection.mutable.ArrayBuffer[Int]()
        var cycles = 0
        while (got.length < n && cycles < 2000) {
          if (getValid) got += getPayload
          tick()
          cycles += 1
        }
        assert(got.length == n, s"$name: collected ${got.length}/$n beats")
        got.toSeq
      }

      // Pass 0 (first, non-final): seed zeros, drain partials
      // io.c.ready stays False: a drain to io.c would stall (trap), not alias.
      firePass(first = true, last = false)
      pushB(bBeats(0))
      dut.io.spillOut.stream.ready #= true
      pushARow(aRowBeats(0, 0))
      val p0r0 = collectN(N, dut.io.spillOut.stream.valid.toBoolean,
        dut.io.spillOut.stream.payload(0).toInt, "spillOut r0")
      pushARow(aRowBeats(0, 1))
      val p0r1 = collectN(N, dut.io.spillOut.stream.valid.toBoolean,
        dut.io.spillOut.stream.payload(0).toInt, "spillOut r1")
      dut.io.spillOut.stream.ready #= false
      val p0 = p0r0 ++ p0r1
      assert(p0 == partialRef(0), s"pass-0 partials $p0 != ${partialRef(0)}")
      settle(2)
      assert(doneEdges == 1, s"expected 1 passDone edge after pass 0, got $doneEdges")

      // Pass 1 (non-first, final): seed pass-0 rows, drain full GEMM
      // spillOut.ready stays False: a drain to spill would stall (trap).
      doneEdges = 0
      firePass(first = false, last = true)
      pushB(bBeats(1))
      pushSpillRow(p0.slice(0, N))
      dut.io.c.stream.ready #= true
      pushARow(aRowBeats(1, 0))
      val cr0 = collectN(N, dut.io.c.stream.valid.toBoolean,
        dut.io.c.stream.payload(0).toInt, "io.c r0")
      pushSpillRow(p0.slice(N, 2 * N))
      pushARow(aRowBeats(1, 1))
      val cr1 = collectN(N, dut.io.c.stream.valid.toBoolean,
        dut.io.c.stream.payload(0).toInt, "io.c r1")
      dut.io.c.stream.ready #= false
      val c = cr0 ++ cr1
      assert(c == fullRef, s"final C $c != $fullRef")
      settle(2)
      assert(doneEdges == 1, s"expected 1 passDone edge after pass 1, got $doneEdges")

      settle(5)
    }
  }

  test("S1 spill order on BF16: bit-exact vs unchanged replica") {
    // Exactly-representable values; the oracle folds K by lanes=2.
    val aVals = Seq(1.0, 0.5, 1.5, 2.0, 0.5, 1.0, 2.0, 1.5,
      2.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 2.0)
    val wVals = Seq(-1.0, -0.5, 0.0, 0.5, 1.0)
    def wAt(k: Int, n: Int): Double = wVals((k + n) % wVals.length)
    val (eW, mW) = (8, 7)
    val aF = aVals.map(fromDouble(_, eW, mW))
    val wF = (0 until N).map(n => (0 until 8).map(k => fromDouble(wAt(k, n), eW, mW)))
    val expectedFull = LayerReplicas.linear(aF, wF, Seq.fill(N)(PZERO), eW, mW, 2)

    // Slice-0 partials with the replica's own chunk order (lanes=2).
    def partialF(p: Int): Seq[F] = {
      val k0 = p * Ks
      (for (m <- 0 until M; n <- 0 until N) yield {
        val c0 = (0 until 2).map(j => fmul(aF(m * 8 + k0 + j), wF(n)(k0 + j), eW, mW))
        val c1 = (0 until 2).map(j => fmul(aF(m * 8 + k0 + 2 + j), wF(n)(k0 + 2 + j), eW, mW))
        fadd(fadd(PZERO, tree(c0, eW, mW), eW, mW), tree(c1, eW, mW), eW, mW)
      }).toSeq
    }

    type Triple = (Boolean, Int, Int)
    def triple(f: F): Triple = (f.s, f.e, f.m)

    SimConfig.withWave.compile(MatmulSpillComp(BF16(), BF16(), M, Ks, N, 2)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      def tick(): Unit = dut.clockDomain.waitSampling()

      def waitReadyOnce(getReady: => Boolean, name: String): Unit = {
        var cycles = 0
        while (!getReady && cycles < 800) { tick(); cycles += 1 }
        assert(getReady, s"$name window never opened")
      }

      def poke(lane: Int, isA: Boolean, f: F): Unit = {
        val sig = if (isA) dut.io.a.stream.payload(lane) else dut.io.b.stream.payload(lane)
        sig.sign #= f.s
        sig.exponent #= f.e
        sig.mantissa #= f.m
      }

      def readTriple(isC: Boolean): Triple = {
        val sig = if (isC) dut.io.c.stream.payload(0) else dut.io.spillOut.stream.payload(0)
        (sig.sign.toBoolean, sig.exponent.toInt, sig.mantissa.toInt)
      }

      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.spillIn.stream.valid #= false
      dut.io.c.stream.ready #= false
      dut.io.spillOut.stream.ready #= false
      dut.io.passFirst #= false
      dut.io.passLast #= false
      dut.io.reArm #= false
      tick()
      tick()

      def firePass(first: Boolean, last: Boolean): Unit = {
        dut.io.passFirst #= first
        dut.io.passLast #= last
        dut.io.reArm #= true
        tick()
        dut.io.reArm #= false
      }

      def bBeatsF(pass: Int): Seq[Seq[F]] = {
        val k0 = pass * Ks
        (0 until N).flatMap(n => Seq(
          Seq(fromDouble(wAt(k0, n), eW, mW), fromDouble(wAt(k0 + 1, n), eW, mW)),
          Seq(fromDouble(wAt(k0 + 2, n), eW, mW), fromDouble(wAt(k0 + 3, n), eW, mW))))
      }

      def aRowBeatsF(pass: Int, m: Int): Seq[Seq[F]] = {
        val k0 = pass * Ks
        Seq(Seq(aF(m * 8 + k0), aF(m * 8 + k0 + 1)), Seq(aF(m * 8 + k0 + 2), aF(m * 8 + k0 + 3)))
      }

      def pushNF(beats: Seq[Seq[F]], setValid: Boolean => Unit, pokeBeat: Seq[F] => Unit,
        getReady: => Boolean, name: String): Unit = {
        require(beats.nonEmpty, s"$name: empty push")
        waitReadyOnce(getReady, name)
        pokeBeat(beats.head)
        tick()
        setValid(true)
        for (i <- beats.indices) {
          tick()
          if (i + 1 < beats.length) pokeBeat(beats(i + 1))
        }
        setValid(false)
        tick()
      }

      def pushBF(beats: Seq[Seq[F]]): Unit = {
        pushNF(beats, v => dut.io.b.stream.valid #= v, beat => {
          poke(0, isA = false, beat(0))
          poke(1, isA = false, beat(1))
        }, dut.io.b.stream.ready.toBoolean, "B")
      }

      def pushARowF(beats: Seq[Seq[F]]): Unit = {
        pushNF(beats, v => dut.io.a.stream.valid #= v, beat => {
          poke(0, isA = true, beat(0))
          poke(1, isA = true, beat(1))
        }, dut.io.a.stream.ready.toBoolean, "A row")
      }

      def pushSpillRowF(vals: Seq[F]): Unit = {
        pushNF(vals.map(f => Seq(f)), v => dut.io.spillIn.stream.valid #= v, beat => {
          dut.io.spillIn.stream.payload(0).sign #= beat(0).s
          dut.io.spillIn.stream.payload(0).exponent #= beat(0).e
          dut.io.spillIn.stream.payload(0).mantissa #= beat(0).m
        }, dut.io.spillIn.stream.ready.toBoolean, "spill seed")
      }

      def collectNTriples(n: Int, isC: Boolean, name: String): Seq[Triple] = {
        val got = scala.collection.mutable.ArrayBuffer[Triple]()
        var cycles = 0
        while (got.length < n && cycles < 2000) {
          val v = if (isC) dut.io.c.stream.valid.toBoolean else dut.io.spillOut.stream.valid.toBoolean
          if (v) got += readTriple(isC)
          tick()
          cycles += 1
        }
        assert(got.length == n, s"$name: collected ${got.length}/$n beats")
        got.toSeq
      }

      // Pass 0: partials must match the replica chunk order
      firePass(first = true, last = false)
      pushBF(bBeatsF(0))
      dut.io.spillOut.stream.ready #= true
      pushARowF(aRowBeatsF(0, 0))
      val p0r0 = collectNTriples(N, isC = false, "spillOut r0")
      pushARowF(aRowBeatsF(0, 1))
      val p0r1 = collectNTriples(N, isC = false, "spillOut r1")
      dut.io.spillOut.stream.ready #= false
      val expP0 = partialF(0).map(triple)
      assert((p0r0 ++ p0r1) == expP0, s"pass-0 partials ${p0r0 ++ p0r1} != $expP0")

      // Pass 1: seed pass-0 rows, drain full result vs replica
      firePass(first = false, last = true)
      pushBF(bBeatsF(1))
      pushSpillRowF(partialF(0).slice(0, N))
      dut.io.c.stream.ready #= true
      pushARowF(aRowBeatsF(1, 0))
      val cr0 = collectNTriples(N, isC = true, "io.c r0")
      pushSpillRowF(partialF(0).slice(N, 2 * N))
      pushARowF(aRowBeatsF(1, 1))
      val cr1 = collectNTriples(N, isC = true, "io.c r1")
      dut.io.c.stream.ready #= false
      val expFull = expectedFull.map(triple)
      assert((cr0 ++ cr1) == expFull, s"final C ${cr0 ++ cr1} != replica $expFull")

      tick()
      tick()
    }
  }

  test("S1 bias-zero gate: BiasAddOp with zero bias is an exact passthrough") {
    // The parking FIFO absorbs all 4 beats, so feed and collect are fully
    // decoupled and sequential; every beat incl. the last is observed.
    SimConfig.withWave.compile(BiasZeroComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      // Same tick discipline as the matmul benches: read a DUT signal before
      // every edge (bare poke-then-edge raced the edge here).
      def tick(): Unit = {
        val _ = dut.io.a.stream.ready.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= false
      dut.io.reArm #= false
      tick()
      tick()

      val aBeats = Seq(Seq(5, -3), Seq(7, 0), Seq(-8, 12), Seq(1, 1))
      // v3: pre-poke + settle tick (valid LOW), valid HIGH held, exactly one
      // tick per beat, next payload poked right after its edge (full cycle
      // to land). No valid toggling, no conditional waits on hot streams.
      // c.ready is armed EARLY (fifo empty: nothing to lose, edges of slack
      // for the poke to land) — never just-in-time, or the first record
      // lands without its transfer and the whole collect shifts by one.
      dut.io.c.stream.ready #= true
      dut.io.b.stream.payload(0) #= 0
      tick()
      tick()
      dut.io.b.stream.valid #= true
      for (_ <- 0 until 4) {
        tick()
      }
      dut.io.b.stream.valid #= false
      tick()

      // A window opens once the bias is loaded (bounded poll, valid LOW).
      var wc = 0
      while (!dut.io.a.stream.ready.toBoolean && wc < 800) {
        tick()
        wc += 1
      }
      assert(dut.io.a.stream.ready.toBoolean, "A window never opened")
      // Interleaved feed+collect: every presented pop beat is recorded in the
      // same iteration (nothing flows unobserved while ready is live).
      val got = scala.collection.mutable.ArrayBuffer[Int]()
      def collectPoll(): Unit = {
        if (dut.io.c.stream.valid.toBoolean) {
          got += dut.io.c.stream.payload(0).toInt
          got += dut.io.c.stream.payload(1).toInt
        }
      }
      dut.io.a.stream.payload(0) #= aBeats(0)(0)
      dut.io.a.stream.payload(1) #= aBeats(0)(1)
      tick()
      dut.io.a.stream.valid #= true
      for ((beat, i) <- aBeats.zipWithIndex) {
        collectPoll()
        tick()
        if (i + 1 < aBeats.length) {
          dut.io.a.stream.payload(0) #= aBeats(i + 1)(0)
          dut.io.a.stream.payload(1) #= aBeats(i + 1)(1)
        }
      }
      dut.io.a.stream.valid #= false
      // Drain the fifo tail (pop latency trails the last push).
      var cycles = 0
      while (got.length < 8 && cycles < 500) {
        collectPoll()
        tick()
        cycles += 1
      }
      dut.io.c.stream.ready #= false
      assert(got.toSeq == aBeats.flatten, s"zero-bias I8 out ${got.toSeq} != in ${aBeats.flatten}")
      tick()
      tick()
    }

    // BF16 leg: exact (sign, exp, mant) triples, not just decoded doubles.
    SimConfig.withWave.compile(BiasZeroComp(BF16())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.a.stream.ready.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.a.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.c.stream.ready #= false
      dut.io.reArm #= false
      tick()
      tick()

      val (eW, mW) = (8, 7)
      val inVals = Seq(1.5, -2.0, 0.5, 0.0, -0.75, 3.25, 1.0, -1.0)
      val inF = inVals.map(fromDouble(_, eW, mW))
      def pokeChunk(c0: F, c1: F): Unit = {
        dut.io.a.stream.payload(0).sign #= c0.s
        dut.io.a.stream.payload(0).exponent #= c0.e
        dut.io.a.stream.payload(0).mantissa #= c0.m
        dut.io.a.stream.payload(1).sign #= c1.s
        dut.io.a.stream.payload(1).exponent #= c1.e
        dut.io.a.stream.payload(1).mantissa #= c1.m
      }
      def pokeZeroBias(): Unit = {
        dut.io.b.stream.payload(0).sign #= false
        dut.io.b.stream.payload(0).exponent #= 0
        dut.io.b.stream.payload(0).mantissa #= 0
      }
      // c.ready armed early (see I8 leg): the fifo is empty, nothing to
      // lose, and the poke lands long before the first pop matters.
      dut.io.c.stream.ready #= true
      pokeZeroBias()
      tick()
      dut.io.b.stream.valid #= true
      for (_ <- 0 until 4) {
        tick()
      }
      dut.io.b.stream.valid #= false
      tick()

      var wc = 0
      while (!dut.io.a.stream.ready.toBoolean && wc < 800) {
        tick()
        wc += 1
      }
      assert(dut.io.a.stream.ready.toBoolean, "A window never opened")
      val got = scala.collection.mutable.ArrayBuffer[(Boolean, Int, Int)]()
      def collectPoll(): Unit = {
        if (dut.io.c.stream.valid.toBoolean) {
          for (lane <- 0 until 2) {
            got += ((dut.io.c.stream.payload(lane).sign.toBoolean,
              dut.io.c.stream.payload(lane).exponent.toInt,
              dut.io.c.stream.payload(lane).mantissa.toInt))
          }
        }
      }
      val chunks = inF.grouped(2).toSeq
      pokeChunk(chunks(0)(0), chunks(0)(1))
      tick()
      dut.io.a.stream.valid #= true
      for ((chunk, i) <- chunks.zipWithIndex) {
        collectPoll()
        tick()
        if (i + 1 < chunks.length) pokeChunk(chunks(i + 1)(0), chunks(i + 1)(1))
      }
      dut.io.a.stream.valid #= false
      // Drain the fifo tail (pop latency trails the last push).
      var cycles = 0
      while (got.length < 8 && cycles < 500) {
        collectPoll()
        tick()
        cycles += 1
      }
      dut.io.c.stream.ready #= false
      val exp = inF.map(f => (f.s, f.e, f.m))
      assert(got.toSeq == exp, s"zero-bias BF16 out ${got.toSeq} != in $exp")
      tick()
      tick()
    }
  }

  test("S1 LinearLayer spill smoke on I8: bias once, y silent on pass 0") {
    val bias = Seq(10, 20, 30)
    val expectedY = fullRef.zipWithIndex.map { case (v, i) => v + bias(i % N) }
    SimConfig.withWave.compile(LinearSpillComp()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      var doneEdges = 0
      var prevDone = false
      var yValidSeen = false
      def tick(): Unit = {
        val cur = dut.io.passDone.toBoolean
        if (cur && !prevDone) doneEdges += 1
        prevDone = cur
        if (dut.io.y.stream.valid.toBoolean) yValidSeen = true
        dut.clockDomain.waitSampling()
      }
      def settle(cycles: Int): Unit = for (_ <- 0 until cycles) tick()

      def waitReadyOnce(getReady: => Boolean, name: String): Unit = {
        var cycles = 0
        while (!getReady && cycles < 800) { tick(); cycles += 1 }
        assert(getReady, s"$name window never opened")
      }

      dut.io.a.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.spillIn.stream.valid #= false
      dut.io.y.stream.ready #= false
      dut.io.spillOut.stream.ready #= false
      dut.io.passFirst #= false
      dut.io.passLast #= false
      dut.io.reArm #= false
      dut.io.biasReArm #= false
      settle(2)

      def firePass(first: Boolean, last: Boolean): Unit = {
        dut.io.passFirst #= first
        dut.io.passLast #= last
        dut.io.reArm #= true
        dut.io.biasReArm #= true
        tick()
        dut.io.reArm #= false
        dut.io.biasReArm #= false
      }

      def pushN(beats: Seq[Seq[Int]], setValid: Boolean => Unit, setPayload: Seq[Int] => Unit,
        getReady: => Boolean, name: String): Unit = {
        require(beats.nonEmpty, s"$name: empty push")
        waitReadyOnce(getReady, name)
        setPayload(beats.head)
        tick()
        setValid(true)
        for (i <- beats.indices) {
          tick()
          if (i + 1 < beats.length) setPayload(beats(i + 1))
        }
        setValid(false)
        tick()
      }

      def pushW(beats: Seq[Seq[Int]]): Unit = {
        pushN(beats, v => dut.io.w.stream.valid #= v, beat => {
          dut.io.w.stream.payload(0) #= beat(0)
          dut.io.w.stream.payload(1) #= beat(1)
        }, dut.io.w.stream.ready.toBoolean, "W")
      }

      def pushARow(beats: Seq[Seq[Int]]): Unit = {
        pushN(beats, v => dut.io.a.stream.valid #= v, beat => {
          dut.io.a.stream.payload(0) #= beat(0)
          dut.io.a.stream.payload(1) #= beat(1)
        }, dut.io.a.stream.ready.toBoolean, "A row")
      }

      def pushSpillRow(vals: Seq[Int]): Unit = {
        pushN(vals.map(v => Seq(v)), v => dut.io.spillIn.stream.valid #= v, beat => {
          dut.io.spillIn.stream.payload(0) #= beat(0)
        }, dut.io.spillIn.stream.ready.toBoolean, "spill seed")
      }

      def collectSpill(n: Int): Seq[Int] = {
        val got = scala.collection.mutable.ArrayBuffer[Int]()
        var cycles = 0
        while (got.length < n && cycles < 2000) {
          if (dut.io.spillOut.stream.valid.toBoolean)
            got += dut.io.spillOut.stream.payload(0).toInt
          tick()
          cycles += 1
        }
        assert(got.length == n, s"spillOut: collected ${got.length}/$n beats")
        got.toSeq
      }

      def collectY(n: Int): Seq[Int] = {
        val got = scala.collection.mutable.ArrayBuffer[Int]()
        var cycles = 0
        while (got.length < n && cycles < 2000) {
          if (dut.io.y.stream.valid.toBoolean)
            got += dut.io.y.stream.payload(0).toInt
          tick()
          cycles += 1
        }
        assert(got.length == n, s"io.y: collected ${got.length}/$n beats")
        got.toSeq
      }

      // Pass 0: garbage on io.b must be IGNORED (zeros selected)
      // io.b.valid=True with 77: the mux must never raise ready.
      firePass(first = true, last = false)
      pushW(bBeats(0))
      dut.io.spillOut.stream.ready #= true
      dut.io.y.stream.ready #= true
      dut.io.b.stream.payload(0) #= 77
      dut.io.b.stream.valid #= true
      pushARow(aRowBeats(0, 0))
      assert(!dut.io.b.stream.ready.toBoolean, "pass 0 consumed io.b: bias-zero mux is wrong")
      val p0r0 = collectSpill(N)
      pushARow(aRowBeats(0, 1))
      val p0r1 = collectSpill(N)
      dut.io.spillOut.stream.ready #= false
      dut.io.b.stream.valid #= false
      val p0 = p0r0 ++ p0r1
      assert(p0 == partialRef(0), s"pass-0 partials $p0 != ${partialRef(0)}")
      assert(!yValidSeen, "io.y must stay silent on a non-final pass (bias not yet applied)")
      settle(20)
      assert(!yValidSeen, "io.y spoke after pass-0 drain")
      assert(!dut.io.b.stream.ready.toBoolean, "pass 0 consumed io.b during settle")
      dut.io.y.stream.ready #= false
      settle(2)
      assert(doneEdges == 1, s"expected 1 passDone edge after pass 0, got $doneEdges")

      // Pass 1: seed pass-0 rows, real bias once, y == full linear + bias
      doneEdges = 0
      yValidSeen = false
      firePass(first = false, last = true)
      pushW(bBeats(1))
      pushSpillRow(p0.slice(0, N))
      // The real bias arrives on the final pass (3 beats, lanes=1), before
      // the matmul output needs it. v3: pre-poke + settle, valid held, one
      // tick per beat.
      waitReadyOnce(dut.io.b.stream.ready.toBoolean, "bias")
      dut.io.b.stream.payload(0) #= bias(0)
      tick()
      dut.io.b.stream.valid #= true
      for ((bv, i) <- bias.zipWithIndex) {
        tick()
        if (i + 1 < bias.length) dut.io.b.stream.payload(0) #= bias(i + 1)
      }
      dut.io.b.stream.valid #= false
      tick()
      dut.io.y.stream.ready #= true
      pushARow(aRowBeats(1, 0))
      val yr0 = collectY(N)
      pushSpillRow(p0.slice(N, 2 * N))
      pushARow(aRowBeats(1, 1))
      val yr1 = collectY(N)
      dut.io.y.stream.ready #= false
      val got = yr0 ++ yr1
      assert(got == expectedY, s"final y $got != $expectedY")
      settle(2)
      assert(doneEdges == 1, s"expected 1 passDone edge after pass 1, got $doneEdges")

      settle(5)
    }
  }
}
