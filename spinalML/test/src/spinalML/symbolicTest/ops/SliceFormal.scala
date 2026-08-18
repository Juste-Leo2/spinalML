package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8
import spinalML.ops.SliceTestComp // Axis 0 (start=1, end=3, len=4, lanes=2)

case class SliceAxis1TestCompFormal[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(4), lanes = 4))
    val c = master(Tensor(dataType, Seq(2), lanes = 2)) // L_out = 2
  }
  io.c <> spinalML.ops.slice(io.a, start = 1, end = 3, axis = 1)
}

class SliceAxis0Formal extends Component {
  val dut = FormalDut(new SliceTestComp(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  // Axis0 is stateful. It drops chunk 0, forwards chunks 1 and 2, drops chunk 3.
  val lanes = dut.io.a.stream.payload.length
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    for(lane <- 0 until lanes) {
      assert(dut.io.c.stream.payload(lane) === dut.io.a.stream.payload(lane), s"SliceAxis0 mismatch on lane $lane")
    }
  }
  
  cover(dut.io.c.stream.valid && dut.io.c.stream.ready)
}

class SliceAxis1Formal extends Component {
  val dut = FormalDut(new SliceAxis1TestCompFormal(I8()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)

  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.a.stream.valid)
  assume(dut.io.c.stream.ready)

  val outLanes = dut.io.c.stream.payload.length
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    for(lane <- 0 until outLanes) {
      assert(dut.io.c.stream.payload(lane) === dut.io.a.stream.payload(1 + lane), s"SliceAxis1 mismatch on lane $lane")
    }
  }
}

object SliceFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(15)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SliceAxis0Formal, "slice_axis0_i8")

    FormalConfig
      .withSymbiYosys
      .withProve(3)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new SliceAxis1Formal, "slice_axis1_i8")
  }
}
