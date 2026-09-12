// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.arithmetic

import spinalML._

/**
 * Configuration for arithmetic multipliers and processing units.
 *
 * @param target  Hardware target specification (Target.FPGA, Target.ASIC, Target.Simulation).
 * @param useDsp  If true, attempts hardware DSP mapping on FPGA. If false, forces soft LUT logic.
 * @param latency Pipeline stage latency (0 = purely combinational, 1 = output registered, 2 = input+output).
 */
case class ArithmeticConfig(
  target: Target = Target.current,
  useDsp: Boolean = {
    val envNoDsp = sys.env.get("SPINALML_NO_DSP").exists(v => v == "1" || v.equalsIgnoreCase("true"))
    val propNoDsp = sys.props.get("spinalml.no_dsp").exists(v => v == "1" || v.equalsIgnoreCase("true") || v.isEmpty)
    !envNoDsp && !propNoDsp
  },
  latency: Int = 1
) {
  require(latency >= 0 && latency <= 2, s"Arithmetic pipeline latency must be 0, 1, or 2, got $latency")

  def isAsic: Boolean = target.isAsic
  def isFpga: Boolean = target.isFpga
  def isSim: Boolean = target.isSim
}

object ArithmeticConfig {
  def default: ArithmeticConfig = ArithmeticConfig()
  def noDsp: ArithmeticConfig = ArithmeticConfig(useDsp = false, latency = 0)
  def combinational: ArithmeticConfig = ArithmeticConfig(latency = 0)
  def pipelined: ArithmeticConfig = ArithmeticConfig(latency = 1)

  /** Explicit configuration targeting an ASIC standard-cell flow (e.g. Sky130 / OpenROAD). */
  def asic(pdk: PdkFamily = PdkFamily.Sky130, latency: Int = 1): ArithmeticConfig =
    ArithmeticConfig(target = Target.ASIC(pdk, pipelineStages = latency), useDsp = false, latency = latency)

  /** Explicit configuration targeting an FPGA vendor architecture. */
  def fpga(family: FpgaFamily = FpgaFamily.Gowin, useDsp: Boolean = true, latency: Int = 1): ArithmeticConfig =
    ArithmeticConfig(target = Target.FPGA(family, useHardDsp = useDsp), useDsp = useDsp, latency = latency)

  /** Explicit configuration for cycle-accurate simulation. */
  def simulation(latency: Int = 1): ArithmeticConfig =
    ArithmeticConfig(target = Target.Simulation, useDsp = false, latency = latency)
}
