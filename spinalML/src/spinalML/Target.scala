// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML

/** Supported FPGA vendor architectures. */
sealed trait FpgaFamily
object FpgaFamily {
  case object Gowin   extends FpgaFamily
  case object Xilinx  extends FpgaFamily
  case object Lattice extends FpgaFamily
  case object Intel   extends FpgaFamily
  case object Generic extends FpgaFamily
}

/** Supported ASIC Process Design Kits (PDKs). */
sealed trait PdkFamily
object PdkFamily {
  case object Sky130    extends PdkFamily
  case object GF180     extends PdkFamily
  case object FreePDK45 extends PdkFamily
  case object Generic   extends PdkFamily
}

/**
 * Universal Hardware Target Specification (3-Layer Architecture).
 *
 * Configures silicon synthesis parameters across FPGA vendor hard macros,
 * ASIC standard-cell pipelining (OpenROAD / OpenLane), and cycle-accurate simulation.
 */
sealed trait Target {
  def isAsic: Boolean = false
  def isFpga: Boolean = false
  def isSim: Boolean = false
}

object Target {

  /** FPGA hardware deployment target. */
  case class FPGA(
    family: FpgaFamily = FpgaFamily.Gowin,
    useHardDsp: Boolean = true
  ) extends Target {
    override def isFpga: Boolean = true
  }

  /** ASIC standard-cell physical design target (e.g. SkyWater 130nm). */
  case class ASIC(
    pdk: PdkFamily = PdkFamily.Sky130,
    pipelineStages: Int = 1
  ) extends Target {
    override def isAsic: Boolean = true
  }

  /** Pure cycle-accurate simulation or symbolic verification target. */
  case object Simulation extends Target {
    override def isSim: Boolean = true
  }

  /** Converts legacy DspTarget into a Target.FPGA. */
  def fromDspTarget(dt: spinalML.dsp.DspTarget, useDsp: Boolean = true): Target = dt match {
    case spinalML.dsp.DspTarget.Gowin   => FPGA(FpgaFamily.Gowin, useHardDsp = useDsp)
    case spinalML.dsp.DspTarget.Xilinx  => FPGA(FpgaFamily.Xilinx, useHardDsp = useDsp)
    case spinalML.dsp.DspTarget.Lattice => FPGA(FpgaFamily.Lattice, useHardDsp = useDsp)
    case spinalML.dsp.DspTarget.Intel   => FPGA(FpgaFamily.Intel, useHardDsp = useDsp)
    case spinalML.dsp.DspTarget.Generic => FPGA(FpgaFamily.Generic, useHardDsp = useDsp)
  }

  /** Resolves a Target from a string name (CLI / Board configuration). */
  def fromString(s: String, useDsp: Boolean = true): Target = {
    s.trim.toLowerCase match {
      case "asic" | "sky130"    => ASIC(PdkFamily.Sky130, pipelineStages = 1)
      case "gf180"              => ASIC(PdkFamily.GF180, pipelineStages = 1)
      case "freepdk45"          => ASIC(PdkFamily.FreePDK45, pipelineStages = 1)
      case "sim" | "simulation" => Simulation
      case "gowin"              => FPGA(FpgaFamily.Gowin, useHardDsp = useDsp)
      case "xilinx"             => FPGA(FpgaFamily.Xilinx, useHardDsp = useDsp)
      case "lattice"            => FPGA(FpgaFamily.Lattice, useHardDsp = useDsp)
      case "intel"              => FPGA(FpgaFamily.Intel, useHardDsp = useDsp)
      case _                    => FPGA(FpgaFamily.Generic, useHardDsp = useDsp)
    }
  }

  /** Resolves the active Target from environment variables or JVM system properties. */
  def current: Target = {
    val targetStr = sys.env.get("SPINALML_TARGET").orElse(sys.props.get("spinalml.target"))
    val noDsp = sys.env.get("SPINALML_NO_DSP").exists(v => v == "1" || v.equalsIgnoreCase("true")) ||
                sys.props.get("spinalml.no_dsp").exists(v => v == "1" || v.equalsIgnoreCase("true") || v.isEmpty)
    targetStr match {
      case Some(s) => fromString(s, useDsp = !noDsp)
      case None    => FPGA(FpgaFamily.Generic, useHardDsp = !noDsp)
    }
  }
}
