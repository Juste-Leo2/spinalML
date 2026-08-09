package spinalML.dtypes

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._

class I8Test extends AnyFunSuite {
  test("I8 bit width") {
    SpinalConfig().generateVerilog(new Component {
      assert(widthOf(I8()) == 8)
    })
  }
}
