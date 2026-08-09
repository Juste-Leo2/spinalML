package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, FP4_E2M1}

class ExpTest extends AnyFunSuite {
  test("Exp LUT compilation on I8") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(I8(), Seq(2), lanes = 2))
      val c = master(Tensor(I8(), Seq(2), lanes = 2))
      c <> exp(a)
    })
  }

  test("Exp LUT compilation on FP8") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP8_E4M3(), Seq(2), lanes = 2))
      val c = master(Tensor(FP8_E4M3(), Seq(2), lanes = 2))
      c <> exp(a)
    })
  }
  
  test("Exp LUT compilation on FP4") {
    SpinalConfig().generateVerilog(new Component {
      val a = slave(Tensor(FP4_E2M1(), Seq(2), lanes = 2))
      val c = master(Tensor(FP4_E2M1(), Seq(2), lanes = 2))
      c <> exp(a)
    })
  }
}
