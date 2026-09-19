// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.I8

class DoubleBufferStreamerTest extends AnyFunSuite {
  test("generate_verilog") {
    // We generate Verilog without renaming ports so Python can map them directly
    SpinalConfig().generateVerilog(DoubleBufferStreamer(I8(), depth = 16, lanes = 1))
  }

  test("BUG-DDR-07: nextTile must NOT fire prematurely under downstream backpressure until tile is fully drained") {
    val depth = 8
    val lanes = 1
    SimConfig.compile(DoubleBufferStreamer(I8(), depth = depth, lanes = lanes)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.tileReady #= false
      dut.io.streamOut.ready #= false // Downstream backpressure!
      dut.io.readData(0) #= 0
      dut.io.reArm #= false
      dut.clockDomain.waitSampling(5)

      // Signal tile ready for tile 1
      dut.io.tileReady #= true
      dut.clockDomain.waitSampling()
      dut.io.tileReady #= false

      // Feed readData during BRAM reads while downstream is completely stalled
      var sawPrematureNextTile = false
      for (cycle <- 0 until 15) {
        dut.io.readData(0) #= ((cycle + 1) & 0x7F)
        dut.clockDomain.waitSampling()
        if (dut.io.nextTile.toBoolean) {
          sawPrematureNextTile = true
        }
      }

      // Assert that nextTile was NOT asserted while downstream was stalled
      assert(!sawPrematureNextTile, "BUG-DDR-07: nextTile pulsed prematurely while streamOut was backpressured (0 elements popped)!")

      // Now release downstream backpressure and consume all elements of tile 1
      dut.io.streamOut.ready #= true
      var wordsPopped = 0
      var nextTileSeenAtWord = -1

      for (cycle <- 0 until 20) {
        if (dut.io.streamOut.valid.toBoolean && dut.io.streamOut.ready.toBoolean) {
          wordsPopped += 1
          if (dut.io.nextTile.toBoolean) {
            nextTileSeenAtWord = wordsPopped
          }
        }
        dut.clockDomain.waitSampling()
      }

      assert(wordsPopped == depth, s"Expected $depth words popped, but got $wordsPopped")
      assert(nextTileSeenAtWord == depth, s"nextTile should only be asserted on the last word ($depth), but was seen at word $nextTileSeenAtWord")
    }
  }
}
