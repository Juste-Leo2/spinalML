// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}

case class LayerNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(16, 4), lanes = 4))
  val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
  val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
  val y = master(Tensor(dataType, Seq(16, 4), lanes = 4))
  
  val comp = LayerNorm1D(dataType, channels = 4, seqLen = 16)
  comp.io.reArm := False
  comp.io.x <> x
  comp.io.gamma <> gamma
  comp.io.beta <> beta
  y <> comp.io.y
}

// LAY-01 repro: two weight generations on the same live datapath. The re-arm
// input restarts the gamma/beta load sequence so generation 2 is accepted.
case class LayerNormReArmTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4, 4), lanes = 4))
    val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
    val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
    val reArm = in Bool()
    val y = master(Tensor(dataType, Seq(4, 4), lanes = 4))
  }

  val comp = LayerNorm1D(dataType, channels = 4, seqLen = 4)
  comp.io.x <> io.x
  comp.io.gamma <> io.gamma
  comp.io.beta <> io.beta
  comp.io.reArm := io.reArm
  io.y <> comp.io.y
}

class LayerNormTest extends AnyFunSuite {
  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"LayerNorm1D compilation on $name") {
      SpinalConfig().generateVerilog(LayerNormTestComp(dt()))
    }
  }

  test("LayerNorm1D re-arm accepts a second generation (LAY-01)") {
    SimConfig.compile(LayerNormReArmTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.io.reArm #= false
      dut.io.gamma.stream.valid #= false
      dut.io.beta.stream.valid #= false
      dut.io.x.stream.valid #= false
      dut.io.y.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      // Bounded readiness poll: used as the generation-2 discriminator
      // (without re-arm the FSM stays in run mode and never accepts gamma).
      def waitReady(ready: => Boolean, label: String, timeout: Int = 2000): Unit = {
        var cycles = 0
        while (!ready && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(ready, s"LAY-01: $label never became ready")
      }

      // MaxPool1DTest idiom: keep `valid` high across the burst and move to the
      // next payload after each ready sampling point (one beat per iteration).
      def sendBurst(valid: Bool, ready: Bool, beats: Int)(setPayload: Int => Unit): Unit = {
        for (b <- 0 until beats) {
          setPayload(b)
          valid #= true
          dut.clockDomain.waitSamplingWhere(ready.toBoolean)
        }
        valid #= false
        dut.clockDomain.waitSampling()
      }

      def sendGamma(values: Seq[Int]): Unit =
        sendBurst(dut.io.gamma.stream.valid, dut.io.gamma.stream.ready, 1) { _ =>
          for (i <- values.indices) dut.io.gamma.stream.payload(i) #= values(i)
        }

      def sendBeta(values: Seq[Int]): Unit =
        sendBurst(dut.io.beta.stream.valid, dut.io.beta.stream.ready, 1) { _ =>
          for (i <- values.indices) dut.io.beta.stream.payload(i) #= values(i)
        }

      def sendX(beats: Int): Unit = {
        for (b <- 0 until beats) {
          for (i <- 0 until 4) dut.io.x.stream.payload(i) #= 1 + b + i
          dut.io.x.stream.valid #= true
          dut.clockDomain.waitSampling()
          var cycles = 0
          while (!dut.io.x.stream.ready.toBoolean && cycles < 1000) {
            dut.clockDomain.waitSampling()
            cycles += 1
        }
          assert(dut.io.x.stream.ready.toBoolean, "LAY-01: x stalled during burst")
        }
        dut.io.x.stream.valid #= false
        dut.clockDomain.waitSampling()
      }

      // Concurrent output collector: the layer streams its frame while the
      // stimulus is being sent, so count continuously and slice per frame.
      var yBeats = 0
      fork {
        while (true) {
          if (dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean) yBeats += 1
          dut.clockDomain.waitSampling()
        }
      }

      def recvY(beats: Int, fromBeat: Int, timeout: Int = 3000): Int = {
        var cycles = 0
        while (yBeats < fromBeat + beats && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        scala.math.min(yBeats - fromBeat, beats)
      }

      // Generation 1
      sendGamma(Seq(2, 3, 4, 5))
      sendBeta(Seq(1, 1, 1, 1))
      sendX(4)
      assert(recvY(4, 0) == 4, "gen1 did not produce a full frame")

      // Command boundary: re-arm, then generation 2 with new coefficients.
      dut.io.reArm #= true
      dut.clockDomain.waitSampling()
      dut.io.reArm #= false

      waitReady(dut.io.gamma.stream.ready.toBoolean, "gen2 gamma")
      sendGamma(Seq(1, 1, 1, 1))
      sendBeta(Seq(2, 2, 2, 2))
      sendX(4)
      assert(recvY(4, 4) == 4, "gen2 did not produce a full frame")

      println("[LayerNormTest] LAY-01: second generation accepted after re-arm")
    }
  }

  test("LayerNorm1D constant frame (LAY-04)") {
    SimConfig.compile(LayerNormTestComp(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.x.stream.valid #= false
      dut.gamma.stream.valid #= false
      dut.beta.stream.valid #= false
      dut.y.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      def waitReady(ready: => Boolean, label: String, timeout: Int = 2000): Unit = {
        var cycles = 0
        while (!ready && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(ready, s"LAY-04: $label never became ready")
      }

      def sendBeat(valid: Bool, ready: Bool)(setPayload: => Unit): Unit = {
        setPayload
        valid #= true
        dut.clockDomain.waitSamplingWhere(ready.toBoolean)
        valid #= false
      }

      waitReady(dut.gamma.stream.ready.toBoolean, "gamma")
      sendBeat(dut.gamma.stream.valid, dut.gamma.stream.ready) {
        for (i <- 0 until 4) dut.gamma.stream.payload(i) #= 1
      }
      sendBeat(dut.beta.stream.valid, dut.beta.stream.ready) {
        for (i <- 0 until 4) dut.beta.stream.payload(i) #= 7
      }

      var yBeats = 0
      val yValues = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
      fork {
        while (true) {
          if (dut.y.stream.valid.toBoolean && dut.y.stream.ready.toBoolean) {
            yValues += (0 until 4).map(i => dut.y.stream.payload(i).toInt)
            yBeats += 1
          }
          dut.clockDomain.waitSampling()
        }
      }

      for (b <- 0 until 16) {
        for (i <- 0 until 4) dut.x.stream.payload(i) #= 3
        dut.x.stream.valid #= true
        dut.clockDomain.waitSampling()
        var cycles = 0
        while (!dut.x.stream.ready.toBoolean && cycles < 1000) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(dut.x.stream.ready.toBoolean, "LAY-04: x stalled during burst")
      }
      dut.x.stream.valid #= false

      waitReady(yBeats >= 16, "constant frame output", 3000)
      for (beat <- yValues.take(16); v <- beat) {
        assert(v == 7, s"LAY-04: constant frame y=$v (expected beta=7)")
      }
      println("[LayerNormTest] LAY-04: constant I8 frame yields beta")
    }
  }

  test("LayerNorm1D constant frame on FP8 (LAY-04)") {
    SimConfig.compile(LayerNormTestComp(FP8_E4M3())).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.x.stream.valid #= false
      dut.gamma.stream.valid #= false
      dut.beta.stream.valid #= false
      dut.y.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      def setFloat(p: spinalML.dtypes.FloatML, exp: Int, mant: Int): Unit = {
        p.sign #= false
        p.exponent #= exp
        p.mantissa #= mant
      }

      def waitReady(ready: => Boolean, label: String, timeout: Int = 2000): Unit = {
        var cycles = 0
        while (!ready && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(ready, s"LAY-04: $label never became ready")
      }

      def sendBeat(valid: Bool, ready: Bool)(setPayload: => Unit): Unit = {
        setPayload
        valid #= true
        dut.clockDomain.waitSamplingWhere(ready.toBoolean)
        valid #= false
      }

      waitReady(dut.gamma.stream.ready.toBoolean, "gamma")
      sendBeat(dut.gamma.stream.valid, dut.gamma.stream.ready) {
        for (i <- 0 until 4) setFloat(dut.gamma.stream.payload(i), 7, 0) // 1.0
      }
      sendBeat(dut.beta.stream.valid, dut.beta.stream.ready) {
        for (i <- 0 until 4) setFloat(dut.beta.stream.payload(i), 8, 0) // 2.0
      }

      var yBeats = 0
      val yFields = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
      fork {
        while (true) {
          if (dut.y.stream.valid.toBoolean && dut.y.stream.ready.toBoolean) {
            yFields += ((dut.y.stream.payload(0).exponent.toInt, dut.y.stream.payload(0).mantissa.toInt))
            yBeats += 1
          }
          dut.clockDomain.waitSampling()
        }
      }

      for (b <- 0 until 16) {
        for (i <- 0 until 4) setFloat(dut.x.stream.payload(i), 9, 0) // 4.0
        dut.x.stream.valid #= true
        dut.clockDomain.waitSampling()
        var cycles = 0
        while (!dut.x.stream.ready.toBoolean && cycles < 1000) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(dut.x.stream.ready.toBoolean, "LAY-04: x stalled during burst")
      }
      dut.x.stream.valid #= false

      waitReady(yBeats >= 16, "constant frame output", 3000)
      for ((exp, mant) <- yFields.take(16)) {
        assert(exp == 8 && mant == 0, s"LAY-04: FP8 constant frame exponent=$exp mantissa=$mant (expected beta=2.0)")
      }
      println("[LayerNormTest] LAY-04: constant FP8 frame yields beta")
    }
  }

  // LAY-04 residual: diff != 0 but diff^2 underflows. Frame [0.5, 0.5625, 0.5, 0.5]
  // (FP8): sum tree -> 2.0, mu = 0.5, diffs = [0, 0.0625, 0, 0], 0.0625^2 = 2^-8
  // flushed to 0 by FTZ -> var = 0. Without a representable epsilon the rsqrt LUT
  // decodes input 0 with the 1/sqrt(x+1e-9) guard, saturates to 448 and yields
  // y = 0.0625 * 448 = 28.0. With eps = min normal 2^-6: invStd = 8 and
  // y = [0, 0.5, 0, 0] (gamma=1, beta=0).
  test("LayerNorm1D small-variance FP8 frame keeps invStd finite (LAY-04 residual)") {
    SimConfig.compile(LayerNormTestComp(FP8_E4M3())).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.x.stream.valid #= false
      dut.gamma.stream.valid #= false
      dut.beta.stream.valid #= false
      dut.y.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      def setFloat(p: spinalML.dtypes.FloatML, exp: Int, mant: Int): Unit = {
        p.sign #= false
        p.exponent #= exp
        p.mantissa #= mant
      }

      def waitReady(ready: => Boolean, label: String, timeout: Int = 2000): Unit = {
        var cycles = 0
        while (!ready && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(ready, s"LAY-04: $label never became ready")
      }

      def sendBeat(valid: Bool, ready: Bool)(setPayload: => Unit): Unit = {
        setPayload
        valid #= true
        dut.clockDomain.waitSamplingWhere(ready.toBoolean)
        valid #= false
      }

      waitReady(dut.gamma.stream.ready.toBoolean, "gamma")
      sendBeat(dut.gamma.stream.valid, dut.gamma.stream.ready) {
        for (i <- 0 until 4) setFloat(dut.gamma.stream.payload(i), 7, 0) // 1.0
      }
      sendBeat(dut.beta.stream.valid, dut.beta.stream.ready) {
        for (i <- 0 until 4) setFloat(dut.beta.stream.payload(i), 0, 0) // 0.0
      }

      val yRows = scala.collection.mutable.ArrayBuffer[Seq[(Int, Int)]]()
      fork {
        while (true) {
          if (dut.y.stream.valid.toBoolean && dut.y.stream.ready.toBoolean) {
            yRows += (0 until 4).map(i =>
              (dut.y.stream.payload(i).exponent.toInt, dut.y.stream.payload(i).mantissa.toInt))
          }
          dut.clockDomain.waitSampling()
        }
      }

      // 0.5 = 2^-1 (exp field 6), 0.5625 = 1.125 * 2^-1 (exp 6, mant 1)
      val row = Seq((6, 0), (6, 1), (6, 0), (6, 0))
      for (b <- 0 until 16) {
        for (i <- 0 until 4) setFloat(dut.x.stream.payload(i), row(i)._1, row(i)._2)
        dut.x.stream.valid #= true
        dut.clockDomain.waitSampling()
        var cycles = 0
        while (!dut.x.stream.ready.toBoolean && cycles < 1000) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(dut.x.stream.ready.toBoolean, "LAY-04: x stalled during burst")
      }
      dut.x.stream.valid #= false

      waitReady(yRows.length >= 16, "small-variance output", 3000)
      val expected = Seq((0, 0), (6, 0), (0, 0), (0, 0))
      for ((beat, bi) <- yRows.take(16).zipWithIndex; (got, lane) <- beat.zipWithIndex) {
        assert(got == expected(lane),
          s"LAY-04 residual: beat $bi lane $lane = (exp=${got._1}, mant=${got._2}) " +
            s"expected (exp=${expected(lane)._1}, mant=${expected(lane)._2}) — invStd saturated?")
      }
      println("[LayerNormTest] LAY-04 residual: small-variance FP8 frame keeps a finite invStd")
    }
  }
}
