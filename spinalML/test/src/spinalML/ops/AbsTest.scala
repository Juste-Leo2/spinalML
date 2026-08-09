package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I4, I16, FP4_E2M1, BF16}

case class AbsTestComp[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 2), lanes = 2))
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 2))
  }
  io.c <> abs(io.a)
}

class AbsTest extends AnyFunSuite {
  test("Abs simulation on I4") {
    SimConfig.withWave.compile(AbsTestComp(I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()
      
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= -3
      dut.io.a.stream.payload(1) #= 2
      
      dut.clockDomain.waitSamplingWhere(dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean)
      
      assert(dut.io.c.stream.payload(0).toInt == 3)
      assert(dut.io.c.stream.payload(1).toInt == 2)
      
      dut.io.a.stream.valid #= false
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Abs compilation on I16") {
    SpinalConfig().generateVerilog(AbsTestComp(I16()))
  }

  test("Abs compilation on FP4") {
    SpinalConfig().generateVerilog(AbsTestComp(FP4_E2M1()))
  }

  test("Abs compilation on BF16") {
    SpinalConfig().generateVerilog(AbsTestComp(BF16()))
  }
}
