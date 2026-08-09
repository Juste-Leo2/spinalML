package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

class ScaleAddTest extends AnyFunSuite {
  test("ScaleAdd compilation on I8") {
    SpinalConfig().generateVerilog(new Component {
      val x = slave(Tensor(I8(), Seq(2), lanes = 2))
      val a = slave(Tensor(I8(), Seq(2), lanes = 2))
      val b = slave(Tensor(I8(), Seq(2), lanes = 2))
      val c = master(Tensor(I8(), Seq(2), lanes = 2))
      c <> scale_add(x, a, b)
    })
  }
}
