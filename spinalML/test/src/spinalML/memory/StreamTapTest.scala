// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinalML.dtypes.I8
import spinalML.tensors.Tensor
import org.scalatest.funsuite.AnyFunSuite

// S2c StreamTap unit harness: 4 I8 elems, 2 lanes (packing proven), 2 beats.
case class StreamTapComp() extends Component {
  val io = new Bundle {
    val streamIn = slave(Tensor(I8(), Seq(2, 2), lanes = 2))
    val streamOut = master(Tensor(I8(), Seq(2, 2), lanes = 2))
    val arm = in Bool()
    val replay = in Bool()
  }
  val tap = StreamTap(I8(), Seq(2, 2), 2)
  tap.io.arm := io.arm
  tap.io.replay := io.replay
  tap.io.streamIn.stream << io.streamIn.stream
  io.streamOut.stream << tap.io.streamOut.stream
}

class StreamTapTest extends AnyFunSuite {

  test("S2c StreamTap: transparent snoop, verbatim replay x2, re-arm") {
    SimConfig.withWave.compile {
      val dut = StreamTapComp()
      dut.setDefinitionName("StreamTapUnitComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.streamOut.stream.valid.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.stream.valid #= false
      dut.io.streamOut.stream.ready #= false
      dut.io.arm #= false
      dut.io.replay #= false
      tick(); tick()

      // Arm (command boundary) then snoop 2 beats, pass-through exact.
      dut.io.arm #= true
      tick()
      dut.io.arm #= false
      val beats = Seq(Seq(5, -3), Seq(7, 0))
      dut.io.streamOut.stream.ready #= true
      dut.io.streamIn.stream.payload(0) #= beats(0)(0)
      dut.io.streamIn.stream.payload(1) #= beats(0)(1)
      tick()
      dut.io.streamIn.stream.valid #= true
      val seen = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
      for ((beat, i) <- beats.zipWithIndex) {
        tick()
        // Pass-through: same-cycle valid + payload.
        assert(dut.io.streamOut.stream.valid.toBoolean, "snoop must pass through")
        seen += Seq(dut.io.streamOut.stream.payload(0).toInt,
          dut.io.streamOut.stream.payload(1).toInt)
        if (i + 1 < beats.length) {
          dut.io.streamIn.stream.payload(0) #= beats(i + 1)(0)
          dut.io.streamIn.stream.payload(1) #= beats(i + 1)(1)
        }
      }
      dut.io.streamIn.stream.valid #= false
      tick()
      assert(seen.toSeq == beats, s"snoop pass-through ${seen.toSeq} != $beats")

      // Replay twice: verbatim, then re-arm resets to snoop.
      for (round <- 0 until 2) {
        dut.io.replay #= true
        tick()
        dut.io.replay #= false
        val got = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
        var cycles = 0
        while (got.length < 2 && cycles < 100) {
          if (dut.io.streamOut.stream.valid.toBoolean)
            got += Seq(dut.io.streamOut.stream.payload(0).toInt,
              dut.io.streamOut.stream.payload(1).toInt)
          tick(); cycles += 1
        }
        assert(got.toSeq == beats, s"replay round $round ${got.toSeq} != $beats")
      }
      // Stream ends after the recorded beats (no wrap-around chatter).
      var extra = 0
      for (_ <- 0 until 10) {
        if (dut.io.streamOut.stream.valid.toBoolean) extra += 1
        tick()
      }
      assert(extra == 0, "replay must go quiet after the recorded beats")

      // Re-arm: back to transparent snoop of new data.
      dut.io.arm #= true
      tick()
      dut.io.arm #= false
      dut.io.streamIn.stream.payload(0) #= 11
      dut.io.streamIn.stream.payload(1) #= 22
      tick()
      dut.io.streamIn.stream.valid #= true
      tick()
      assert(dut.io.streamOut.stream.valid.toBoolean, "re-armed tap must snoop")
      assert(dut.io.streamOut.stream.payload(0).toInt == 11)
      assert(dut.io.streamOut.stream.payload(1).toInt == 22)
      dut.io.streamIn.stream.valid #= false
      tick(); tick()
    }
  }

  test("S2c StreamTap: replay backpressures, upstream held off") {
    SimConfig.withWave.compile {
      val dut = StreamTapComp()
      dut.setDefinitionName("StreamTapBackpressureComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      def tick(): Unit = {
        val _ = dut.io.streamOut.stream.valid.toBoolean
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.stream.valid #= false
      dut.io.streamOut.stream.ready #= true
      dut.io.arm #= false
      dut.io.replay #= false
      tick(); tick()

      // Snoop one full pass first (exactly one tick per beat).
      dut.io.arm #= true
      tick()
      dut.io.arm #= false
      dut.io.streamIn.stream.payload(0) #= 1
      dut.io.streamIn.stream.payload(1) #= 2
      tick()
      dut.io.streamIn.stream.valid #= true
      tick()
      dut.io.streamIn.stream.payload(0) #= 3
      dut.io.streamIn.stream.payload(1) #= 4
      tick()
      dut.io.streamIn.stream.valid #= false
      tick()

      // Replay with downstream stalled: valid held, first beat stable.
      dut.io.replay #= true
      tick()
      dut.io.replay #= false
      dut.io.streamOut.stream.ready #= false
      tick(); tick()
      assert(dut.io.streamOut.stream.valid.toBoolean, "replay valid must hold under stall")
      assert(dut.io.streamOut.stream.payload(0).toInt == 1)
      // Upstream is held off in replay mode (loud, not aliased).
      assert(!dut.io.streamIn.stream.ready.toBoolean, "upstream must stall in replay")
      // Release: both beats flow in order, then quiet. One settle tick so
      // the ready poke has landed (S1 lag rule) — the first sample after a
      // ready transition is stale (no fire that cycle).
      dut.io.streamOut.stream.ready #= true
      tick()
      val got = scala.collection.mutable.ArrayBuffer[Int]()
      var cycles = 0
      while (got.length < 4 && cycles < 100) {
        if (dut.io.streamOut.stream.valid.toBoolean) {
          got += dut.io.streamOut.stream.payload(0).toInt
          got += dut.io.streamOut.stream.payload(1).toInt
        }
        tick(); cycles += 1
      }
      assert(got.toSeq == Seq(1, 2, 3, 4), s"ordered replay ${got.toSeq}")
      tick(); tick()
    }
  }
}
