// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.{I8, I16}

import scala.util.Random

class LineBuffer2DTest extends AnyFunSuite {

  test("LineBuffer2D: exact delay line of depth D with continuous valid stream") {
    val depth = 5
    val numInputs = 30

    SimConfig.withVerilator.compile(LineBuffer2D(I8(), depth)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.io.push.valid #= false
      dut.io.push.payload #= 0
      dut.clockDomain.waitSampling()

      val testValues = Seq.tabulate(numInputs)(i => ((i * 3 + 7) & 0x7F) - 64)
      val received = scala.collection.mutable.ArrayBuffer[Int]()

      var inIdx = 0
      var outIdx = 0

      while (outIdx < numInputs) {
        if (inIdx < numInputs) {
          dut.io.push.valid #= true
          dut.io.push.payload #= testValues(inIdx)
        } else {
          dut.io.push.valid #= true // keep clocking valid to flush
          dut.io.push.payload #= 0
        }

        dut.clockDomain.waitSampling()

        if (dut.io.pop.valid.toBoolean) {
          received += dut.io.pop.payload.toInt
          outIdx += 1
        }
        if (inIdx < numInputs) inIdx += 1
      }

      // Since pop.valid asserts 1 cycle after push.valid, exactly (depth - 1)
      // initial dummy zero beats are observed before the first pushed element.
      val expected = Seq.fill(depth - 1)(0) ++ testValues.take(numInputs - (depth - 1))
      assert(received.take(numInputs) == expected, s"Received $received != expected $expected")
    }
  }

  test("LineBuffer2D: robust against arbitrary valid pauses/stalls") {
    val depth = 4
    val numInputs = 25
    val rnd = new Random(1234)

    SimConfig.withVerilator.compile(LineBuffer2D(I16(), depth)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.io.push.valid #= false
      dut.io.push.payload #= 0
      dut.clockDomain.waitSampling()

      val testValues = Seq.tabulate(numInputs)(i => 100 + i * 5)
      val received = scala.collection.mutable.ArrayBuffer[Int]()

      var inIdx = 0

      while (received.length < numInputs) {
        val pushActive = inIdx < numInputs && rnd.nextBoolean()

        dut.io.push.valid #= pushActive
        if (pushActive) {
          dut.io.push.payload #= testValues(inIdx)
        } else if (inIdx >= numInputs) {
          // Flush with valid cycles
          dut.io.push.valid #= true
          dut.io.push.payload #= 0
        }

        dut.clockDomain.waitSampling()

        if (dut.io.pop.valid.toBoolean) {
          received += dut.io.pop.payload.toInt
        }
        if (pushActive) {
          inIdx += 1
        }
      }

      val expected = Seq.fill(depth - 1)(0) ++ testValues.take(numInputs - (depth - 1))
      assert(received.take(numInputs) == expected, s"Received $received != expected $expected under stalls")
    }
  }

  test("LineBuffer2D: edge case depth = 1 behaves as a single register delay") {
    SimConfig.withVerilator.compile(LineBuffer2D(I8(), 1)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.io.push.valid #= false
      dut.clockDomain.waitSampling()

      val values = Seq(11, 22, 33, 44, 55)
      val received = scala.collection.mutable.ArrayBuffer[Int]()

      for (v <- values) {
        dut.io.push.valid #= true
        dut.io.push.payload #= v
        dut.clockDomain.waitSampling()
        if (dut.io.pop.valid.toBoolean) {
          received += dut.io.pop.payload.toInt
        }
      }
      // One more cycle to pop last
      dut.io.push.valid #= true
      dut.io.push.payload #= 0
      dut.clockDomain.waitSampling()
      if (dut.io.pop.valid.toBoolean) {
        received += dut.io.pop.payload.toInt
      }

      val expected = Seq(11, 22, 33, 44, 55)
      assert(received.take(5) == expected)
    }
  }
}
