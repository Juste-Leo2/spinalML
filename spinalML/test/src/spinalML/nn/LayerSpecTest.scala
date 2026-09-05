// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinalML.dtypes.BF16

class LayerSpecTest extends AnyFunSuite {

  test("Linear getOutShape preserves leading rows (features-last)") {
    assert(Linear(4, 3).getOutShape(Seq(2, 4)) == Seq(2, 3))
    assert(Linear(64, 10).getOutShape(Seq(1, 64)) == Seq(1, 10))
    assert(Linear(64, 10).getOutShape(Seq(3, 2, 64)) == Seq(3, 2, 10))
    assert(Linear(8, 4).getOutShape(Seq(5, 6, 8)) == Seq(5, 6, 4))
  }

  test("Linear getOutShape rejects mismatched feature dimension") {
    intercept[IllegalArgumentException](Linear(5, 3).getOutShape(Seq(2, 4)))
    intercept[IllegalArgumentException](Linear(5, 3).getOutShape(Seq(7)))
  }

  test("Flatten produces features-last vector") {
    assert(Flatten().getOutShape(Seq(8, 8)) == Seq(1, 64))
    assert(Flatten().getOutShape(Seq(6, 6, 1)) == Seq(1, 36))
    assert(Flatten().getOutShape(Seq(4)) == Seq(1, 4))
  }

  test("Flatten -> Linear chain shapes") {
    val shape = Flatten().getOutShape(Seq(6, 6, 4))
    assert(shape == Seq(1, 144))
    assert(Linear(144, 10).getOutShape(shape) == Seq(1, 10))
  }

  test("Shape-only specs are identity") {
    for (shape <- Seq(Seq(4), Seq(2, 3), Seq(2, 3, 4))) {
      assert(ReLU().getOutShape(shape) == shape)
      assert(Sigmoid().getOutShape(shape) == shape)
      assert(Tanh().getOutShape(shape) == shape)
      assert(Cast(BF16()).getOutShape(shape) == shape)
    }
  }
}
