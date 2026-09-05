// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core.SpinalConfig
import spinal.lib.bus.amba4.axi.Axi4Config

class HighLevel2DTemplateTest extends AnyFunSuite {
  test("HighLevel2DTemplate compilation") {
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    SpinalConfig().generateVerilog(HighLevel2DTemplate(axiConfig))
  }
}
