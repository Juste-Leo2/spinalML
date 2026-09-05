// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.dtypes

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._

class FP4Test extends AnyFunSuite {
  test("FP4_E2M1 bit width") {
    SpinalConfig().generateVerilog(new Component {
      val t = FP4_E2M1()
      assert(t.expBits == 2)
      assert(t.mantBits == 1)
      assert(widthOf(t) == 4)
    })
  }
}
