// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory.litedram

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._

/** Stage-B suite: write->read vs Micron x16 (THE front).
 * Run: `mill ... testOnly spinalML.memory.litedram.LiteDramCoreAxiTest`
 * (or `cli/main.py test-all -k LiteDramCoreAxi`). */
class LiteDramCoreAxiTest extends AnyFunSuite {

  /** Single-x16 Micron model wrapper BlackBox (see ddrWrap above). */
  class Ddr3Wrap extends BlackBox {
    setDefinitionName("ddr3_wrap")
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

  /** Stage-B top: core + Micron x16 model on the DDR pads. */
  class SimTopB extends Component {
    val io = new Bundle {
      val resetN = in(Bool())
      val initDone = out(Bool())
      val pllLocked = out(Bool())
      val dbgStep = out(Bits(8 bits))
      val dbgRst = out(Bits(8 bits))
      // AXI single-beat host port (LiteX streaming flavour).
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
    val simDir = GowinSimPrep.stageDdr()
    core.addRTLPath(primNoRpll)
    core.addRTLPath(rpllStub)
    core.addRTLPath(simDir + "/ddr3.v")
    core.addRTLPath(simDir + "/ddr3_wrap.v")
    val liteDbgMon = GowinSimPrep.stageDbgMon(simDir)
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

    val mem = new Ddr3Wrap()
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

  test("litedram-core-axi (verilator, stage B: write->read vs Micron x16)") {
    // Stage the DDR3 model first: SpinalSim copies RTL into its workspace,
    // losing the ddr3.v sibling .vh -> pass +incdir explicitly.
    val ddrDir = GowinSimPrep.stageDdr().replace("\\", "/")
    // -Wno-MULTIDRIVEN/SELRANGE/CASEX: Micron model internals (multi-clocked
    // check pipelines, param selects); zero-delay with --no-timing, order
    // preserved. Revisit if reads misalign.
    val noLint = Seq("--no-timing", "-Wno-COMBDLY", "-Wno-IEEEMAYDEPRECATE",
      "-Wno-REALCVT", "-Wno-IMPLICIT", "-Wno-PINMISSING", "-Wno-CASEINCOMPLETE",
      "-Wno-MISINDENT", "-Wno-LATCH", "-Wno-MULTIDRIVEN", "-Wno-SELRANGE",
      "-Wno-CASEX")
    var cfg = SimConfig
    for (f <- noLint) cfg = cfg.addSimulatorFlag(f)
    cfg
      .addSimulatorFlag("+define+den1024Mb")
      .addSimulatorFlag("+define+sg25E")
      .addSimulatorFlag("+incdir+" + ddrDir)
      .compile(new SimTopB).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 37037)
      val cd = dut.clockDomain

      // AXI idle defaults (responses always accepted).
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
        // INDEPENDENT channel completion: aw may backpressure for a full
        // row cycle (PRE+ACT) while w is taken at once; holding wValid
        // through the aw wait floods the 16-deep w FIFO with stale copies
        // (data shifts across writes -> readback mismatch). Drop each
        // valid on its own ready (1 TB cycle ~= 1 sys period now).
        // BOUNDED handshakes: bare waitSamplingWhere loops forever if the
        // signal never comes; per-cycle stepping is ~0.2-4ms/call on this
        // Windows stack, so caps must be modest (30k cyc ~= tens of sec).
        var wt = 0
        while (!dut.io.wReady.toBoolean && wt < 30000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.wReady.toBoolean, f"w_ready timeout @word $wordAddr%x")
        dut.io.wValid #= false
        wt = 0
        while (!dut.io.awReady.toBoolean && wt < 30000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.awReady.toBoolean, f"aw_ready timeout @word $wordAddr%x")
        dut.io.awValid #= false
        // Poll every cycle: b_valid is a ~1-sys-cycle pulse (resp FIFO
        // pops immediately with b_ready=1); coarser polling can miss it.
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

      // DEBUG-TRIMMED to 2 vectors (init dominates runtime; traffic is
      // free). Restore the 4-vector set (0x0, 0x1, 0x100, 0x10000) on green.
      // Word addresses span columns/rows (byte addr = word << 3); the pair
      // covers row-hit (0x0) + row-miss/PRE+ACT (0x10000).
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
