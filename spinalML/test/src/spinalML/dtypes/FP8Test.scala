package spinalML.dtypes

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._

class FP8Test extends AnyFunSuite {
  test("FP8_E4M3 bit width") {
    val t = FP8_E4M3()
    assert(t.expBits == 4)
    assert(t.mantBits == 3)
    assert(widthOf(t) == 8)
  }
  
  test("FP8_E5M2 bit width") {
    val t = FP8_E5M2()
    assert(t.expBits == 5)
    assert(t.mantBits == 2)
    assert(widthOf(t) == 8)
  }
}
