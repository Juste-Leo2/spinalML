// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory.litedram

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._

/** Homegrown DFI responder BlackBox (see dfiResp above). Same pads as the
 * Micron wrapper; dm is read-only here but kept inout/Analog so the
 * Spinal wiring is identical (a Verilog input on a shared wire never
 * contends). */
class DfiResp extends BlackBox {
  setDefinitionName("dfi_resp")
  val io = new Bundle {
    val rst_n = in Bool()
    val ck = in Bool()
    val ck_n = in Bool()
    val cke = in Bool()
    val cs_n = in Bool()
    val ras_n = in Bool()
    val cas_n = in Bool()
    val we_n = in Bool()
    val dm = inout(Analog(Bits(2 bits)))
    val ba = in Bits(3 bits)
    val addr = in Bits(13 bits)
    val dq = inout(Analog(Bits(16 bits)))
    val dqs_p = inout(Analog(Bits(2 bits)))
    val odt = in Bool()
  }
  noIoPrefix()
}

/** Stage-D suite: core + DFI responder (fast digital-path proof, no Micron
 * model weight/noise). Same TB as stage B.
 * Run: `mill ... testOnly spinalML.memory.litedram.LiteDramCoreDfiTest`. */
class LiteDramCoreDfiTest extends AnyFunSuite {

  /** Stage-D top: core + dfi_resp on the DDR pads. */
  class SimTopD extends Component {
    val io = new Bundle {
      val resetN = in(Bool())
      val initDone = out(Bool())
      val pllLocked = out(Bool())
      val dbgStep = out(Bits(8 bits))
      val dbgRst = out(Bits(8 bits))
      val awValid = in Bool()
      val awAddr = in UInt(27 bits)
      val wValid = in Bool()
      val wData = in Bits(64 bits)
      val wStrb = in Bits(8 bits)
      val arValid = in Bool()
      val arAddr = in UInt(27 bits)
      val awReady = out Bool()
      val wReady = out Bool()
      val bValid = out Bool()
      val bResp = out Bits(2 bits)
      val arReady = out Bool()
      val rValid = out Bool()
      val rData = out Bits(64 bits)
    }
    val core = new LiteDramCore()
    val (primNoRpll, rpllStub) = GowinSimPrep.prepare()
    val simDir = new java.io.File(new java.io.File(LiteDramCore.corePath).getParent, "sim")
    simDir.mkdirs()
    core.addRTLPath(primNoRpll)
    core.addRTLPath(rpllStub)
    core.addRTLPath(GowinSimPrep.stageResp(simDir.getAbsolutePath))
    val liteDbgMon = GowinSimPrep.stageDbgMon(simDir.getAbsolutePath)
    core.addRTLPath(liteDbgMon)
    core.mapClockDomain(clock = core.io.clk27)
    core.io.reset_n := io.resetN
    io.initDone := core.io.init_done
    io.pllLocked := core.io.pll_locked
    io.dbgStep := core.io.dbg_step
    io.dbgRst := core.io.dbg_rst

    // AXI host side: single-beat INCR x64, TB-driven valid/addr/data.
    core.io.aw_valid := io.awValid
    core.io.aw_payload_addr := io.awAddr
    core.io.aw_first := True
    core.io.aw_last := True
    core.io.aw_payload_burst := 1
    core.io.aw_payload_len := 0
    core.io.aw_payload_size := 3
    core.io.aw_payload_lock := False
    core.io.aw_payload_prot := 0
    core.io.aw_payload_cache := 0
    core.io.aw_payload_qos := 0
    core.io.aw_payload_region := 0
    core.io.aw_param_id := 0
    core.io.aw_param_dest := False
    core.io.aw_param_user := False
    core.io.w_valid := io.wValid
    core.io.w_payload_data := io.wData
    core.io.w_payload_strb := io.wStrb
    core.io.w_first := True
    core.io.w_last := True
    core.io.w_param_id := 0
    core.io.w_param_dest := False
    core.io.w_param_user := False
    core.io.b_ready := True
    core.io.ar_valid := io.arValid
    core.io.ar_payload_addr := io.arAddr
    core.io.ar_first := True
    core.io.ar_last := True
    core.io.ar_payload_burst := 1
    core.io.ar_payload_len := 0
    core.io.ar_payload_size := 3
    core.io.ar_payload_lock := False
    core.io.ar_payload_prot := 0
    core.io.ar_payload_cache := 0
    core.io.ar_payload_qos := 0
    core.io.ar_payload_region := 0
    core.io.ar_param_id := 0
    core.io.ar_param_dest := False
    core.io.ar_param_user := False
    core.io.r_ready := True
    io.awReady := core.io.aw_ready
    io.wReady := core.io.w_ready
    io.bValid := core.io.b_valid
    io.bResp := core.io.b_payload_resp
    io.arReady := core.io.ar_ready
    io.rValid := core.io.r_valid
    io.rData := core.io.r_payload_data

    val mem = new DfiResp()
    mem.io.rst_n := core.io.reset_n_1
    mem.io.ck := core.io.clk_p
    mem.io.ck_n := core.io.clk_n
    mem.io.cke := core.io.cke
    mem.io.cs_n := core.io.cs_n
    mem.io.ras_n := core.io.ras_n
    mem.io.cas_n := core.io.cas_n
    mem.io.we_n := core.io.we_n
    val dmW = Analog(Bits(2 bits))
    dmW := core.io.dm
    mem.io.dm := dmW
    mem.io.ba := core.io.ba
    mem.io.addr := core.io.a(12 downto 0)
    val dqW = Analog(Bits(16 bits))
    val dqsW = Analog(Bits(2 bits))
    core.io.dq := dqW
    mem.io.dq := dqW
    core.io.dqs_p := dqsW
    mem.io.dqs_p := dqsW
    mem.io.odt := core.io.odt
  }

  test("litedram-core-dfi (verilator, stage D: write->read vs DFI responder)") {
    var cfg = SimConfig
      .addSimulatorFlag("--no-timing")
      .addSimulatorFlag("-Wno-COMBDLY")
      .addSimulatorFlag("-Wno-IEEEMAYDEPRECATE")
      .addSimulatorFlag("-Wno-REALCVT")
      .addSimulatorFlag("-Wno-IMPLICIT")
      .addSimulatorFlag("-Wno-PINMISSING")
      .addSimulatorFlag("-Wno-CASEINCOMPLETE")
      .addSimulatorFlag("-Wno-MISINDENT")
      .addSimulatorFlag("-Wno-LATCH")
      .addSimulatorFlag("-Wno-MULTIDRIVEN")
      .addSimulatorFlag("-Wno-SELRANGE")
      .addSimulatorFlag("-Wno-CASEX")
    cfg.compile(new SimTopD).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 37037)
      val cd = dut.clockDomain

      dut.io.awValid #= false
      dut.io.wValid #= false
      dut.io.arValid #= false

      dut.io.resetN #= false
      cd.waitSampling(10)
      dut.io.resetN #= true

      var waited = 0
      while (!dut.io.pllLocked.toBoolean && waited < 100000) {
        cd.waitSampling(1000)
        waited += 1000
      }
      assert(dut.io.pllLocked.toBoolean, "rPLL never locked in sim")
      waited = 0
      while (!dut.io.initDone.toBoolean && waited < 500000) {
        cd.waitSampling(1000)
        waited += 1000
      }
      assert(dut.io.initDone.toBoolean,
        f"init_done stuck 0 (dbg_step=0x${dut.io.dbgStep.toBigInt}%02X)")

      def axiWrite(wordAddr: BigInt, data: BigInt): Unit = {
        dut.io.awAddr #= wordAddr
        dut.io.wData #= data
        dut.io.wStrb #= 0xFF
        dut.io.awValid #= true
        dut.io.wValid #= true
        // BOUNDED handshakes (see stage B): never waitSamplingWhere
        // unbounded, never spin 200k roundtrips (~15min on this stack).
        var wt = 0
        while (!dut.io.wReady.toBoolean && wt < 30000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.wReady.toBoolean, f"w_ready timeout @word $wordAddr%x")
        dut.io.wValid #= false
        wt = 0
        while (!dut.io.awReady.toBoolean && wt < 30000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.awReady.toBoolean, f"aw_ready timeout @word $wordAddr%x")
        dut.io.awValid #= false
        var n = 0
        while (!dut.io.bValid.toBoolean && n < 30000) { cd.waitSampling(1); n += 1 }
        assert(dut.io.bValid.toBoolean, f"write b_valid timeout @word $wordAddr%x")
        assert(dut.io.bResp.toBigInt == 0, "write RESP not OKAY")
        cd.waitSampling(5)
      }

      def axiRead(wordAddr: BigInt): BigInt = {
        dut.io.arAddr #= wordAddr
        dut.io.arValid #= true
        var wt = 0
        while (!dut.io.arReady.toBoolean && wt < 30000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.arReady.toBoolean, f"ar_ready timeout @word $wordAddr%x")
        dut.io.arValid #= false
        var n = 0
        while (!dut.io.rValid.toBoolean && n < 30000) { cd.waitSampling(1); n += 1 }
        assert(dut.io.rValid.toBoolean, f"read r_valid timeout @word $wordAddr%x (PHY read hang?)")
        val d = dut.io.rData.toBigInt
        cd.waitSampling(5)
        d
      }

      val vectors = Seq(
        (BigInt(0x0), BigInt("0123456789ABCDEF", 16)),
        (BigInt(0x10000), BigInt("DEADBEEFCAFEBABE", 16))
      )
      for ((wa, data) <- vectors) axiWrite(wa, data)
      for ((wa, exp) <- vectors) {
        val got = axiRead(wa)
        assert(got == exp, f"mismatch @word $wa%x: got $got%016X exp $exp%016X")
      }
    }
  }
}
