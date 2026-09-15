// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.arithmetic

import spinal.core._
import spinalML._
import spinalML.dtypes.FloatML
import spinalML.utils.Float
import spinalML.primitives.gowin.GowinMULT18X18

/**
 * Universal Hardware Multiplier (Layer 2 - Dual-Target: FPGA & ASIC).
 *
 * Implements hardware arithmetic with target-aware silicon mapping:
 * - Target.ASIC: Pure behavioral RTL with clean register retiming stages (0, 1, or 2 cycles).
 *   Strictly avoids proprietary vendor FPGA blackboxes to ensure zero-defect synthesis
 *   in open-source ASIC physical flows (OpenROAD, SkyWater 130nm, GF180).
 * - Target.FPGA (Gowin): Maps to the clocked MULT18X18 hard DSP block when running on
 *   physical silicon (with OUT_REG=1 to avoid the upstream silicon signedness flaw).
 * - Target.FPGA (Generic/Xilinx/Lattice/Intel): Synthesizes clean behavioral multipliers
 *   transparently inferred as DSP48/DSP blocks by vendor synthesis tools.
 * - Target.FPGA (--no-dsp / useHardDsp=false): Synthesizes entirely into soft LUT logic.
 * - Target.Simulation: Ultra-fast cycle-accurate behavioral execution.
 */
object HardwareMul {

  /**
   * Multiplies two hardware signals with an explicit accumulator return type.
   */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    enable: Bool,
    accType: HardType[TAcc],
    latency: Int,
    config: ArithmeticConfig
  ): TAcc = {
    require(latency >= 0 && latency <= 2, s"Latency must be 0, 1, or 2, got $latency")

    val isSim = GenerationFlags.simulation.isEnabled || config.target.isSim
    val isGowinPhysical = config.target match {
      case Target.FPGA(FpgaFamily.Gowin, useHardDsp) => useHardDsp && config.useDsp && !isSim
      case _                                         => false
    }

    (a, b) match {
      case (valA: SInt, valB: SInt) =>
        val wA = valA.getBitsWidth
        val wB = valB.getBitsWidth
        val wOut = widthOf(accType)

        if (isGowinPhysical && latency >= 1 && wA <= 18 && wB <= 18) {
          // Physical Gowin MULT18X18 mapping with clocked registers
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
          // Standard behavioral RTL (ASIC standard-cells, Simulation, Generic FPGA DSP inference, or soft LUT)
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

        if (isGowinPhysical && latency >= 1 && wA <= 18 && wB <= 18) {
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
        throw new IllegalArgumentException(
          s"Unsupported operand types for HardwareMul: (${a.getClass.getSimpleName}, ${b.getClass.getSimpleName})"
        )
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
    config: ArithmeticConfig
  ): T = apply(a, b, enable, HardType(a), latency, config)

  /**
   * Convenience overload with default enable = True and default ArithmeticConfig.
   */
  def apply[T <: Data, TAcc <: Data](
    a: T,
    b: T,
    accType: HardType[TAcc]
  ): TAcc = apply(a, b, enable = True, accType = accType, latency = 1, config = ArithmeticConfig.default)
}
