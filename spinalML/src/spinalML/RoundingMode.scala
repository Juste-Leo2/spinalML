// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML

/**
 * Rounding policy for narrowing operations (Wave 4 `SemanticsRounding`).
 *
 *  - [[Rne]] (default): round-to-nearest-even, unbiased, aligned with
 *    numpy/PyTorch. See `docs/rounding_policy.md`.
 *  - [[Truncate]]: legacy truncation, bit-exact with the pre-Wave-4 hardware.
 *
 * Elaboration-only parameter: in [[Truncate]] mode the RNE datapath is not
 * elaborated, so 0 LUT is added. It must never become a runtime signal.
 */
sealed trait RoundingMode

object RoundingMode {
  case object Rne extends RoundingMode
  case object Truncate extends RoundingMode
}

object RoundingConfig {

  /** Resolves a RoundingMode from a string (CLI / env / JVM property). */
  def fromString(s: String): RoundingMode = {
    s.trim.toLowerCase match {
      case "trunc" | "truncate" | "floor" => RoundingMode.Truncate
      case _                              => RoundingMode.Rne
    }
  }

  /**
   * Resolves the active RoundingMode from environment variables or JVM
   * system properties. Mirrors `Target.current`.
   *
   * Precedence: explicit per-op parameter > SPINALML_ROUNDING /
   * -Dspinalml.rounding > default RNE.
   */
  def current: RoundingMode = {
    val roundingStr = sys.env.get("SPINALML_ROUNDING").orElse(sys.props.get("spinalml.rounding"))
    roundingStr match {
      case Some(s) => fromString(s)
      case None    => RoundingMode.Rne
    }
  }
}
