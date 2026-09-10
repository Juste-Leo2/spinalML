// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dsp

import spinal.core._
import spinalML.dtypes.FloatML
import spinalML.utils.Float

/**
 * Universal Hardware DSP Multiplier.
 *
 * Provides a unified, portable, and bit-exact multiplier abstraction for all FPGA targets:
 * - Generic: Behavioral RTL with configurable pipelining, automatically inferred as DSP
 *   primitives by AMD/Xilinx Vivado (DSP48), Intel Quartus, and Lattice tools.
 * - Gowin: Maps to clocked Gowin MULT18X18 hardware macro with OUT_REG=1, eliminating
 *   the physical silicon signedness bug present in unclocked combinational DSP mode.
 * - Fallback LUT: If useDsp=false (e.g. via --no-dsp), synthesizes purely into logic slices.
 *
 * Latency modes:
 * - 0: Purely combinational (maps to LUT logic to ensure silicon safety).
 * - 1: Output registered (OUT_REG=1 / PREG=1) - 1 cycle compute latency.
 * - 2: Input + Output registered (AREG=1, BREG=1, OUT_REG=1) - 2 cycles compute latency.
 */
object DspMul {

  /**
   * Multiplies two hardware signals with explicit accumulator return type.
   */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    enable: Bool,
    accType: HardType[TAcc],
    latency: Int,
    dspConfig: DspConfig
  ): TAcc = {
    require(latency >= 0 && latency <= 2, s"Latency must be 0, 1, or 2, got $latency")

    val isSim = GenerationFlags.simulation.isEnabled || dspConfig.target == DspTarget.Generic
    val isGowinTarget = dspConfig.target == DspTarget.Gowin && !isSim && dspConfig.useDsp

    (a, b) match {
      case (valA: SInt, valB: SInt) =>
        val wA = valA.getBitsWidth
        val wB = valB.getBitsWidth
        val wOut = widthOf(accType)

        if (isGowinTarget && latency >= 1 && wA <= 18 && wB <= 18) {
          // Gowin physical hardware mapping with clocked output register (eliminates Apycula IRBY bug)
          val dsp = new GowinMULT18X18(
            areg = latency >= 2,
            breg = latency >= 2,
            outReg = true,
            pipeReg = false,
            asignReg = latency >= 2,
            bsignReg = latency >= 2
          )
          dsp.io.A := valA.resize(18 bits).asBits
          dsp.io.B := valB.resize(18 bits).asBits
          dsp.io.SIA := B(0, 18 bits)
          dsp.io.SIB := B(0, 18 bits)
          dsp.io.ASIGN := True
          dsp.io.BSIGN := True
          dsp.io.ASEL := False
          dsp.io.BSEL := False
          dsp.io.CE := enable

          val outSigned = dsp.io.DOUT.asSInt.resize(wOut)
          outSigned.asInstanceOf[TAcc]
        } else {
          // Generic behavioral inference (Vivado, Quartus, Simulation, or LUT fallback)
          val rawProd: SInt = latency match {
            case 0 => (valA * valB).resize(wOut)
            case 1 =>
              val p = (valA * valB).resize(wOut)
              RegNextWhen(p, enable, init = p.getZero)
            case 2 =>
              val inA = RegNextWhen(valA, enable, init = valA.getZero)
              val inB = RegNextWhen(valB, enable, init = valB.getZero)
              val p = (inA * inB).resize(wOut)
              RegNextWhen(p, enable, init = p.getZero)
          }
          rawProd.asInstanceOf[TAcc]
        }

      case (valA: UInt, valB: UInt) =>
        val wA = valA.getBitsWidth
        val wB = valB.getBitsWidth
        val wOut = widthOf(accType)

        if (isGowinTarget && latency >= 1 && wA <= 18 && wB <= 18) {
          val dsp = new GowinMULT18X18(
            areg = latency >= 2,
            breg = latency >= 2,
            outReg = true,
            pipeReg = false,
            asignReg = latency >= 2,
            bsignReg = latency >= 2
          )
          dsp.io.A := valA.resize(18 bits).asBits
          dsp.io.B := valB.resize(18 bits).asBits
          dsp.io.SIA := B(0, 18 bits)
          dsp.io.SIB := B(0, 18 bits)
          dsp.io.ASIGN := False
          dsp.io.BSIGN := False
          dsp.io.ASEL := False
          dsp.io.BSEL := False
          dsp.io.CE := enable

          val outUnsigned = dsp.io.DOUT.asUInt.resize(wOut)
          outUnsigned.asInstanceOf[TAcc]
        } else {
          val rawProd: UInt = latency match {
            case 0 => (valA * valB).resize(wOut)
            case 1 =>
              val p = (valA * valB).resize(wOut)
              RegNextWhen(p, enable, init = p.getZero)
            case 2 =>
              val inA = RegNextWhen(valA, enable, init = valA.getZero)
              val inB = RegNextWhen(valB, enable, init = valB.getZero)
              val p = (inA * inB).resize(wOut)
              RegNextWhen(p, enable, init = p.getZero)
          }
          rawProd.asInstanceOf[TAcc]
        }

      case (valA: FloatML, valB: FloatML) =>
        val fProd = Float.mul(valA, valB)
        val outFloat = latency match {
          case 0 => fProd
          case 1 => RegNextWhen(fProd, enable)
          case 2 =>
            val inA = RegNextWhen(valA, enable)
            val inB = RegNextWhen(valB, enable)
            RegNextWhen(Float.mul(inA, inB), enable)
        }
        outFloat.asInstanceOf[TAcc]

      case _ =>
        throw new IllegalArgumentException(s"Unsupported operand types for DspMul: (${a.getClass.getSimpleName}, ${b.getClass.getSimpleName})")
    }
  }

  /**
   * Overload inferring output accumulator type from input operand type.
   */
  def apply[T <: Data](
    a: T,
    b: T,
    enable: Bool,
    latency: Int,
    dspConfig: DspConfig
  ): T = apply(a, b, enable, HardType(a), latency, dspConfig)

  /**
   * Convenience overload with default enable = True and default DspConfig.
   */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    accType: HardType[TAcc]
  ): TAcc = apply(a, b, enable = True, accType = accType, latency = 1, dspConfig = DspConfig.default)
}
