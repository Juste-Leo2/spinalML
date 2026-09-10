// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dsp

/**
 * Target FPGA architecture for DSP block mapping.
 *
 * - Generic: Standard behavioral RTL (RegNextWhen). Transparently inferred as
 *            hardware DSPs by vendor tools (Vivado DSP48, Quartus DSP blocks,
 *            nextpnr-ecp5 sysDSP) and used for fast cycle-accurate C++ simulation.
 * - Gowin:   Targets Gowin GW1N/GW2A/GW5A architectures. Instantiates clocked
 *            MULT18X18 primitives in hardware synthesis to overcome the upstream
 *            Yosys combinational signedness bug on physical silicon.
 * - Xilinx:  Explicit Xilinx DSP48 target (maps to Generic behavioral inference).
 * - Lattice: Explicit Lattice target (maps to Generic behavioral inference).
 * - Intel:   Explicit Intel/Altera target (maps to Generic behavioral inference).
 */
sealed trait DspTarget

object DspTarget {
  case object Generic extends DspTarget
  case object Gowin   extends DspTarget
  case object Xilinx  extends DspTarget
  case object Lattice extends DspTarget
  case object Intel   extends DspTarget

  def fromString(s: String): DspTarget = s.trim.toLowerCase match {
    case "gowin"   => Gowin
    case "xilinx"  => Xilinx
    case "lattice" => Lattice
    case "intel"   => Intel
    case _         => Generic
  }

  /** Resolves the current target platform from environment or system properties. */
  def current: DspTarget = {
    sys.env.get("SPINALML_TARGET")
      .orElse(sys.props.get("spinalml.target"))
      .map(fromString)
      .getOrElse(Generic)
  }
}
