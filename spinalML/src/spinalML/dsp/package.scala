// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML

package object dsp {
  type Target = spinalML.Target
  val Target = spinalML.Target

  type FpgaFamily = spinalML.FpgaFamily
  val FpgaFamily = spinalML.FpgaFamily

  type PdkFamily = spinalML.PdkFamily
  val PdkFamily = spinalML.PdkFamily

  type DspConfig = spinalML.arithmetic.ArithmeticConfig
  val DspConfig = spinalML.arithmetic.ArithmeticConfig

  type GowinMULT18X18 = spinalML.primitives.gowin.GowinMULT18X18
}
