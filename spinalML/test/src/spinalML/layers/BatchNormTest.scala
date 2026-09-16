// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I8, I16, FP8_E4M3, BF16}

case class BatchNormTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val x = slave(Tensor(dataType, Seq(16, 4), lanes = 4))
  val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
  val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
  val y = master(Tensor(dataType, Seq(16, 4), lanes = 4))
  
  y <> batchnorm(x, gamma, beta)
}

// LAY-01 repro: two weight generations on the same live datapath. The re-arm
// input restarts the gamma/beta load sequence so generation 2 is accepted.
case class BatchNormReArmTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(4, 4), lanes = 4))
    val gamma = slave(Tensor(dataType, Seq(4), lanes = 4))
    val beta = slave(Tensor(dataType, Seq(4), lanes = 4))
    val reArm = in Bool()
    val y = master(Tensor(dataType, Seq(4, 4), lanes = 4))
  }

  io.y <> batchnorm(io.x, io.gamma, io.beta, reArm = Some(io.reArm))
}

class BatchNormTest extends AnyFunSuite {
  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"BatchNorm1D compilation on $name") {
      SpinalConfig().generateVerilog(BatchNormTestComp(dt()))
    }
  }

  test("BatchNorm1D re-arm accepts a second generation (LAY-01)") {
    SimConfig.compile(BatchNormReArmTestComp(I8())).doSim { dut =>
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

      def sendX(value: Int, beats: Int): Unit = {
        for (_ <- 0 until beats) {
          for (i <- 0 until 4) dut.io.x.stream.payload(i) #= value
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
      // stimulus is being sent, so capture continuously and slice per frame.
      val yCollected = scala.collection.mutable.ArrayBuffer[Int]()
      fork {
        while (true) {
          if (dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean) {
            for (i <- 0 until 4) yCollected += dut.io.y.stream.payload(i).toInt
          }
          dut.clockDomain.waitSampling()
        }
      }

      def recvY(beats: Int, fromBeat: Int, timeout: Int = 3000): Seq[Int] = {
        var cycles = 0
        while (yCollected.length < (fromBeat + beats) * 4 && cycles < timeout) {
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        yCollected.slice(fromBeat * 4, (fromBeat + beats) * 4).toSeq
      }

      // Generation 1: y = x * gamma + beta
      sendGamma(Seq(2, 3, 4, 5))
      sendBeta(Seq(1, 1, 1, 1))
      sendX(1, 4)
      val y1 = recvY(4, 0)
      assert(y1 == Seq(3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6),
        s"gen1 mismatch: $y1")

      // Command boundary: re-arm, then generation 2 with new coefficients.
      dut.io.reArm #= true
      dut.clockDomain.waitSampling()
      dut.io.reArm #= false

      waitReady(dut.io.gamma.stream.ready.toBoolean, "gen2 gamma")
      sendGamma(Seq(1, 1, 1, 1))
      sendBeta(Seq(2, 2, 2, 2))
      sendX(1, 4)
      val y2 = recvY(4, 4)
      assert(y2 == Seq.fill(16)(3), s"gen2 mismatch: $y2")

      println("[BatchNormTest] LAY-01: second generation accepted after re-arm")
    }
  }
}
