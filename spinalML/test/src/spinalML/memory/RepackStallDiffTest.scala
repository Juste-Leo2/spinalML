// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.dtypes.{BF16, I8}
import spinalML.ops.{RepackOp, repack}
import spinalML.tensors.Tensor

/**
 * M1 étape A (docs/open-mysteries.md): differential stall harness for the
 * dual-mode lane gearbox (ops/repack.scala).
 *
 * Unlike the earlier eager micro-probes (and the original RepackFormal whose
 * assumes pin valid/ready HIGH), this suite drives INDEPENDENT random
 * valid/ready streams, runs TWO strictly serialized back-to-back commands per
 * session (command boundary + reArm + isEmpty exposure), and checks the output
 * bit-stream against an absolute golden: element j must leave as the
 * (j mod lanesOut)-th slice of group j/lanesOut, whatever pacing the
 * environment chooses.
 *
 * Closure criterion of M1: this file is a PERMANENT regression test.
 */
class RepackStallDiffTest extends AnyFunSuite {

  /** DUT wrapping repack() with raw-Bits interfaces so the testbench can
    * drive/read any dtype generically (lane 0 = least significant slice). */
  case class RepackDiffDut(dataType: HardType[Data], totalElements: Int,
                            lanesIn: Int, lanesOut: Int, withFlush: Boolean) extends Component {
    val w = dataType().getBitsWidth
    val io = new Bundle {
      val a = slave(Stream(Bits(lanesIn * w bits)))
      val c = master(Stream(Bits(lanesOut * w bits)))
      val reArm = in Bool()
      val isEmpty = out Bool()
    }

    val inTensor = Tensor(dataType, Seq(totalElements), lanesIn)
    inTensor.stream.valid := io.a.valid
    for ((lane, slice) <- inTensor.stream.payload zip io.a.payload.subdivideIn(lanesIn slices)) {
      lane.assignFromBits(slice)
    }

    val createdOps = scala.collection.mutable.ArrayBuffer[RepackOp[_]]()
    val outTensor = repack(inTensor, lanesOut,
      reArm = Some(io.reArm), created = createdOps, withFlush = withFlush)
    // repack's internal <> already drives inTensor.stream.ready from the
    // gearbox chain — READ it here (writing it twice would be an overlap).
    io.a.ready := inTensor.stream.ready

    io.c.valid := outTensor.stream.valid
    outTensor.stream.ready := io.c.ready
    // lane 0 at the least significant slice, mirroring the input mapping
    io.c.payload := outTensor.stream.payload.map(_.asBits).reverse.reduce(_ ## _)

    io.isEmpty := (if (createdOps.isEmpty) True
                   else createdOps.map(_.io.isEmpty).reduce(_ && _))
  }

  /** One simulation session: `nCmds` strictly serialized back-to-back
    * commands under independent seeded random valid/ready; absolute golden
    * checked on every fired output group. `w` must equal dataType bit width
    * (passed explicitly: HardType.apply needs an elaboration context). */
  def runSession(name: String, dataType: HardType[Data], w: Int, totalElements: Int,
                 lanesIn: Int, lanesOut: Int, withFlush: Boolean,
                 pValid: Double, pReady: Double, seed: Int, nCmds: Int = 2): Unit = {
    val modulus = BigInt(1) << w
    require(totalElements % lanesIn == 0 && totalElements % lanesOut == 0)
    val beatsIn = totalElements / lanesIn
    val groupsOut = totalElements / lanesOut
    val maxCycles = 400000

    val compiled = SimConfig.withVerilator.compile(
      RepackDiffDut(dataType, totalElements, lanesIn, lanesOut, withFlush))

    compiled.doSim(name) { dut =>
      val rng = new scala.util.Random(seed)

      // Element j of command `cmd` carries value (cmd*totalElements + j) mod 2^w,
      // lane l occupying bits [l*w +: w] of the beat word (LSB-first slices).
      def word(elemBase: BigInt, lanes: Int, first: Int): BigInt =
        (0 until lanes).map(l => (((elemBase + first + l) % modulus) << (l * w))).sum

      var sent = 0       // input beats fired, all commands
      var recv = 0       // output groups fired, all commands
      var prevCmd = -1
      var pendingBoundaryCheck = false
      var done = false

      dut.io.a.valid #= false
      dut.io.a.payload #= 0
      dut.io.c.ready #= false
      dut.io.reArm #= false
      dut.clockDomain.forkStimulus(10)

      dut.clockDomain.onSamplings {
        if (!done) {
          // ---- 1) observe the transfers committed at this edge ----
          if (dut.io.a.valid.toBoolean && dut.io.a.ready.toBoolean) sent += 1
          if (dut.io.c.valid.toBoolean && dut.io.c.ready.toBoolean) {
            val cmd = recv / groupsOut
            val gLocal = recv - cmd * groupsOut
            val expect = word(BigInt(cmd) * totalElements, lanesOut, gLocal * lanesOut)
            val actual = dut.io.c.payload.toBigInt
            assert(actual == expect,
              s"$name: group $gLocal of command $cmd corrupted\n" +
                s"  got  0x${actual.toString(16)}\n" +
                s"  want 0x${expect.toString(16)}")
            recv += 1
          }

          // ---- 2) prepare the drivers for the next edge ----
          if (recv >= nCmds * groupsOut) {
            done = true
            dut.io.a.valid #= false
            dut.io.c.ready #= false
            dut.io.reArm #= false
          } else {
            val cmd = recv / groupsOut
            // isEmpty is sampled one edge AFTER the boundary: the gearbox's
            // full flag clears on the clock edge that ends the last group,
            // and pre-edge reads still show the old value.
            var reArmPulse = false
            if (pendingBoundaryCheck) {
              assert(dut.io.isEmpty.toBoolean,
                s"$name: gearbox not empty entering command $cmd (stale partial group)")
              reArmPulse = true
              pendingBoundaryCheck = false
            }
            if (cmd != prevCmd && cmd > 0 && withFlush) pendingBoundaryCheck = true
            prevCmd = cmd

            dut.io.reArm #= reArmPulse
            // Strict flush discipline mirroring the real system (empty-gated
            // cmd.ready, data well after cmd.fire): no beat may be accepted
            // between boundary detection and the reArm pulse — the gearbox's
            // last-wins `when(io.reArm)` would drop a simultaneously
            // captured word.
            val issuing = !reArmPulse && !pendingBoundaryCheck &&
              sent < (cmd + 1) * beatsIn
            dut.io.a.valid #= issuing && rng.nextDouble() < pValid
            if (issuing) {
              val b = sent - cmd * beatsIn
              dut.io.a.payload #= word(BigInt(cmd) * totalElements, lanesIn, b * lanesIn)
            }
            dut.io.c.ready #= rng.nextDouble() < pReady
          }
        }
      }

      // Main thread: ride along until the monitor reports completion.
      var guard = 0
      while (!done && guard < maxCycles) {
        dut.clockDomain.waitSampling()
        guard += 1
      }
      assert(done,
        s"$name: timeout after $guard cycles (sent=$sent/$nCmds*$beatsIn, recv=$recv/$nCmds*$groupsOut)")
      println(f"[PASS] $name%-38s cycles=$guard%6d cmds=$nCmds groups=${recv}%4d " +
        f"pValid=$pValid%.2f pReady=$pReady%.2f seed=$seed flush=$withFlush")
    }
  }

  // ------------------------------------------------------------------
  // The exact ResidualMLP-image configuration that failed during the
  // re-arm session (BF16 4→1 split), now under aggressive stalls:
  // ------------------------------------------------------------------
  test("M1-A split BF16 4-to-1 flushable - random stalls") {
    runSession("bf16_4to1_p50_p50", BF16(), 16, 256, 4, 1, withFlush = true, 0.5, 0.5, seed = 7)
  }
  test("M1-A split BF16 4-to-1 flushable - hot producer slow consumer") {
    runSession("bf16_4to1_p100_p15", BF16(), 16, 256, 4, 1, withFlush = true, 1.0, 0.15, seed = 11)
  }

  test("M1-A split I8 4-to-1 flushable - random stalls") {
    runSession("i8_4to1_p50_p50", I8(), 8, 256, 4, 1, withFlush = true, 0.5, 0.5, seed = 3)
  }
  test("M1-A split I8 16-to-4 flushable - random stalls") {
    runSession("i8_16to4_p50_p50", I8(), 8, 512, 16, 4, withFlush = true, 0.5, 0.5, seed = 5)
  }

  test("M1-A aggregate I8 2-to-4 flushable - random stalls") {
    runSession("i8_2to4_p50_p50", I8(), 8, 64, 2, 4, withFlush = true, 0.5, 0.5, seed = 9)
  }
  test("M1-A aggregate I8 2-to-4 flushable - hot producer") {
    runSession("i8_2to4_p100_p25", I8(), 8, 64, 2, 4, withFlush = true, 1.0, 0.25, seed = 13)
  }

  // The EXACT ResidualMLP-Dense1 suspect shape: lanes-in = 1 => m = 4.
  // Earlier coverage stopped at m = 2 (and m = 3 in the formal chain) — m = 4
  // was the blind spot until the network bisection pointed here.
  test("M1-C aggregate I8 1-to-4 flushable - random stalls (suspect m=4)") {
    runSession("i8_1to4_p50_p50", I8(), 8, 32, 1, 4, withFlush = true, 0.5, 0.5, seed = 23)
    runSession("i8_1to4_p100_p15", I8(), 8, 32, 1, 4, withFlush = true, 1.0, 0.15, seed = 29)
  }

  // The non-multiple W4A8 weight ratio: TWO chained gearboxes via lanes=1.
  test("M1-A chain I8 16-1-25 flushable - random stalls") {
    runSession("i8_chain16_25_p50_p50", I8(), 8, 400, 16, 25, withFlush = true, 0.5, 0.5, seed = 17)
  }
  test("M1-A chain I8 16-1-25 flushable - slow consumer") {
    runSession("i8_chain16_25_p100_p10", I8(), 8, 400, 16, 25, withFlush = true, 1.0, 0.10, seed = 19)
  }

  // Legacy adapter cross-checks (multiple ratios only — the legacy
  // StreamWidthAdapter refuses non-multiple down-conversion without padding).
  test("M1-A cross-check legacy adapter survives the same storms") {
    runSession("bf16_4to1_LEGACY_p50_p50", BF16(), 16, 256, 4, 1, withFlush = false, 0.5, 0.5, seed = 7)
    runSession("i8_2to4_LEGACY_p50_p50", I8(), 8, 64, 2, 4, withFlush = false, 0.5, 0.5, seed = 9)
  }
}
