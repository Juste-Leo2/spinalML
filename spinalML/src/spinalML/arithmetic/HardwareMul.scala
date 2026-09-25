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

  private[arithmetic] def useGowinDsp(config: ArithmeticConfig, isSim: Boolean): Boolean =
    config.target match {
      case Target.FPGA(FpgaFamily.Gowin, useHardDsp) => useHardDsp && config.useDsp && !isSim
      case _                                         => false
    }

  // Physical Gowin MULT18X18 mapping with clocked registers
  private[arithmetic] def gowinDsp(
    a: Bits,
    b: Bits,
    signed: Boolean,
    latency: Int,
    enable: Bool
  ): Bits = {
    val dsp = new GowinMULT18X18(
      areg = latency >= 2,
      breg = latency >= 2,
      outReg = true,
      pipeReg = false,
      asignReg = latency >= 2,
      bsignReg = latency >= 2
    )
    dsp.io.A := a
    dsp.io.B := b
    dsp.io.SIA := B(0, 18 bits)
    dsp.io.SIB := B(0, 18 bits)
    dsp.io.ASIGN := (if (signed) True else False)
    dsp.io.BSIGN := (if (signed) True else False)
    dsp.io.ASEL := False
    dsp.io.BSEL := False
    dsp.io.CE := enable
    dsp.io.DOUT
  }

  /** Staging register with deterministic reset (integer datapaths). */
  private[arithmetic] def stagedInit[T <: Data](x: T, en: Bool): T =
    RegNextWhen(x, en, init = x.getZero)

  /** Staging register without reset value (float datapaths, as before). */
  private[arithmetic] def staged[T <: Data](x: T, en: Bool): T =
    RegNextWhen(x, en)

  /** Shared 0/1/2-cycle retiming: make builds the product, stage registers it. */
  private[arithmetic] def retime[T <: Data](
    a: T,
    b: T,
    enable: Bool,
    latency: Int,
    stage: (T, Bool) => T
  )(make: (T, T) => T): T = latency match {
    case 0 => make(a, b)
    case 1 => stage(make(a, b), enable)
    case 2 => stage(make(stage(a, enable), stage(b, enable)), enable)
  }

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
    val isGowinPhysical = useGowinDsp(config, isSim)

    (a, b) match {
      case (valA: SInt, valB: SInt) =>
        val wOut = widthOf(accType)
        if (isGowinPhysical && latency >= 1 && valA.getBitsWidth <= 18 && valB.getBitsWidth <= 18) {
          gowinDsp(valA.resize(18 bits).asBits, valB.resize(18 bits).asBits,
            signed = true, latency, enable).asSInt.resize(wOut).asInstanceOf[TAcc]
        } else {
          // Standard behavioral RTL (ASIC standard-cells, Simulation, Generic FPGA DSP inference, or soft LUT)
          retime(valA, valB, enable, latency, stagedInit[SInt] _) { (x, y) =>
            (x * y).resize(wOut)
          }.asInstanceOf[TAcc]
        }

      case (valA: UInt, valB: UInt) =>
        val wOut = widthOf(accType)
        if (isGowinPhysical && latency >= 1 && valA.getBitsWidth <= 18 && valB.getBitsWidth <= 18) {
          gowinDsp(valA.resize(18 bits).asBits, valB.resize(18 bits).asBits,
            signed = false, latency, enable).asUInt.resize(wOut).asInstanceOf[TAcc]
        } else {
          // Standard behavioral RTL (ASIC standard-cells, Simulation, Generic FPGA DSP inference, or soft LUT)
          retime(valA, valB, enable, latency, stagedInit[UInt] _) { (x, y) =>
            (x * y).resize(wOut)
          }.asInstanceOf[TAcc]
        }

      case (valA: FloatML, valB: FloatML) =>
        retime(valA, valB, enable, latency, staged[FloatML] _) { (x, y) =>
          Float.mul(x, y)
        }.asInstanceOf[TAcc]

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
