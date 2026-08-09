package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, FP4_E2M1}

class RsqrtTest extends AnyFunSuite {
  test("Rsqrt LUT compilation on FP8") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP8_E4M3(), Seq(2), lanes = 2))
      val c = master(Tensor(FP8_E4M3(), Seq(2), lanes = 2))
      c <> rsqrt(a)
    })
  }
}
