package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.ops.CumsumTestComp
import spinalML.dtypes.{I8, FP4_E2M1, FloatML}

abstract class CumsumFormalBase[T <: Data](dataType: HardType[T]) extends Component {
  val dut = FormalDut(CumsumTestComp(dataType))

  anyseq(dut.io.in.stream.valid)
  anyseq(dut.io.in.stream.payload)
  anyseq(dut.io.out.stream.ready)

  assumeInitial(clockDomain.isResetActive)

  val pastValid = past(dut.io.in.stream.valid)
  val pastReady = past(dut.io.in.stream.ready)
  val pastPayload = past(dut.io.in.stream.payload)
  when(pastValid && !pastReady) {
    assume(dut.io.in.stream.valid)
    assume(dut.io.in.stream.payload === pastPayload)
  }

  val counter = Counter(3)
  val fire = dut.io.in.stream.valid && dut.io.in.stream.ready
  when(fire) { counter.increment() }
  assume(counter.value =/= 3 || !dut.io.in.stream.valid)

  val track = RegInit(False)
  val hasChecked = RegInit(False)

  when(dut.io.out.stream.valid && dut.io.out.stream.ready) {
    track := True
  }

  when(track && !hasChecked) {
    assert(dut.io.out.stream.valid || dut.io.out.stream.ready, "Flow control drop")
    hasChecked := True
  }
}

class CumsumFormal_I8 extends CumsumFormalBase(I8())
class CumsumFormal_FP4 extends CumsumFormalBase(FP4_E2M1())

object CumsumFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new CumsumFormal_I8, "cumsum_i8")

    FormalConfig
      .withSymbiYosys
      .withBMC(4)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new CumsumFormal_FP4, "cumsum_fp4")
  }
}
