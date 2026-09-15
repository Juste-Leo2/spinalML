// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.primitives.gowin

import spinal.core._

/**
 * SpinalHDL BlackBox wrapper for Gowin GW1N/GW2A/GW5A hard DSP macro (MULT18X18).
 *
 * Layer 3 Physical Silicon Primitive.
 * Provides physical hardware mapping for signed/unsigned 18x18 multiplication.
 * To avoid the physical Gowin silicon signedness bug identified in open-source
 * bitstream flows, OUT_REG is configured to 1 (clocked mode).
 */
class GowinMULT18X18(
  areg: Boolean = false,
  breg: Boolean = false,
  outReg: Boolean = true,
  pipeReg: Boolean = false,
  asignReg: Boolean = false,
  bsignReg: Boolean = false,
  resetMode: String = "SYNC"
) extends BlackBox {
  addGeneric("AREG", if (areg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("BREG", if (breg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("OUT_REG", if (outReg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("PIPE_REG", if (pipeReg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("ASIGN_REG", if (asignReg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("BSIGN_REG", if (bsignReg) B(1, 1 bits) else B(0, 1 bits))
  addGeneric("SOA_REG", B(0, 1 bits))
  addGeneric("MULT_RESET_MODE", resetMode)

  val io = new Bundle {
    val A     = in Bits(18 bits)
    val SIA   = in Bits(18 bits)
    val B     = in Bits(18 bits)
    val SIB   = in Bits(18 bits)
    val ASIGN = in Bool()
    val BSIGN = in Bool()
    val ASEL  = in Bool()
    val BSEL  = in Bool()
    val CE    = in Bool()
    val CLK   = in Bool()
    val RESET = in Bool()
    val DOUT  = out Bits(36 bits)
    val SOA   = out Bits(18 bits)
    val SOB   = out Bits(18 bits)
  }

  noIoPrefix()
  mapClockDomain(clock = io.CLK, reset = io.RESET)
  setDefinitionName("MULT18X18")
}
