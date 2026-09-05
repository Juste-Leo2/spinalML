// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinalML.dtypes.I8

class DoubleBufferStreamerTest extends AnyFunSuite {
  test("generate_verilog") {
    // We generate Verilog without renaming ports so Python can map them directly
    SpinalConfig().generateVerilog(DoubleBufferStreamer(I8(), depth = 16, lanes = 1))
  }
}
