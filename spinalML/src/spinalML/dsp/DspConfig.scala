// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dsp

/**
 * Configuration for DSP arithmetic operations.
 *
 * @param target  Target FPGA family (defaults to automatic environment detection).
 * @param useDsp  If true, attempts hardware DSP mapping. If false, forces soft LUT logic.
 * @param latency Pipeline stage latency (0 = purely combinational, 1 = output registered, 2 = input+output).
 */
case class DspConfig(
  target: DspTarget = DspTarget.current,
  useDsp: Boolean = {
    val envNoDsp = sys.env.get("SPINALML_NO_DSP").exists(v => v == "1" || v.equalsIgnoreCase("true"))
    val propNoDsp = sys.props.get("spinalml.no_dsp").exists(v => v == "1" || v.equalsIgnoreCase("true") || v.isEmpty)
    !envNoDsp && !propNoDsp
  },
  latency: Int = 1
) {
  require(latency >= 0 && latency <= 2, s"DSP pipeline latency must be 0, 1, or 2, got $latency")
}

object DspConfig {
  def default: DspConfig = DspConfig()
  def noDsp: DspConfig = DspConfig(useDsp = false, latency = 0)
  def combinational: DspConfig = DspConfig(latency = 0)
  def pipelined: DspConfig = DspConfig(latency = 1)
}
