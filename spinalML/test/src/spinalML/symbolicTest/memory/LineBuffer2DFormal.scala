// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.symbolicTest.memory

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.memory.LineBuffer2D

/**
 * Formal verification for LineBuffer2D using SymbiYosys and CVC4 SMT-BMC.
 *
 * LineBuffer2D delays an accepted stream by exactly `depth` beats using a circular
 * synchronous memory (Mem.readSync) without flip-flop shift register bloat.
 *
 * Properties verified:
 *  1. Valid Latency: pop.valid is strictly equal to RegNext(push.valid).
 *  2. Bit-exact FIFO delay: whenever pop.valid is high, popped payload bit-exactly
 *     matches the accepted push payload from `depth` accepted cycles prior.
 *  3. Stall invariance: arbitrary gaps/bubbles in push.valid preserve data integrity.
 *  4. Non-vacuity covers: buffer wrap-around and stall recovery are fully reachable.
 */
class LineBuffer2DFormal extends Component {
  val clk = in Bool()
  val rst = in Bool()

  val formalCd = ClockDomain(clock = clk)
  val dutCd    = ClockDomain(clock = clk, reset = rst)

  // Enforce power-on reset constraint at the top clock domain (unconditioned by rst)
  new ClockingArea(formalCd) {
    assumeInitial(rst)
    val pastValid = Reg(Bool())
    assumeInitial(!pastValid)
    pastValid := True
    when(pastValid) {
      assume(!rst)
    }
  }

  val depth = 2
  val dut = new ClockingArea(dutCd) {
    val comp = FormalDut(LineBuffer2D(UInt(2 bits), depth = depth))
  }

  // Drive inputs
  anyseq(dut.comp.io.push.valid)
  anyseq(dut.comp.io.push.payload)

  // Golden model in dutCd
  val refQueue = new ClockingArea(dutCd) {
    val q = Vec(Reg(UInt(2 bits)) init (0), depth)
    when(dut.comp.io.push.valid) {
      for (i <- depth - 1 downto 1) {
        q(i) := q(i - 1)
      }
      q(0) := dut.comp.io.push.payload
    }
  }

  new ClockingArea(dutCd) {
    when(pastValid() && !rst) {
      // 1. Valid handshake latency: pop.valid strictly mirrors push.valid delayed by 1 cycle
      assert(dut.comp.io.pop.valid === past(dut.comp.io.push.valid), "pop.valid must match past(push.valid)")

      // 2. Exact FIFO queue ordering: whenever pop is valid, payload bit-exactly matches reference delay line
      when(dut.comp.io.pop.valid) {
        assert(dut.comp.io.pop.payload === refQueue.q(depth - 1), "Popped data does not match reference delay line")
      }
    }

    // Cover properties to ensure reachability and non-vacuity
    cover(dut.comp.io.pop.valid)
    cover(dut.comp.io.pop.valid && dut.comp.io.push.valid)
    cover(pastValid() && past(dut.comp.io.pop.valid) && !dut.comp.io.pop.valid)
  }
}

object LineBuffer2DFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(6)
      .withTimeout(60)
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new LineBuffer2DFormal, "line_buffer_2d_formal")
  }
}

