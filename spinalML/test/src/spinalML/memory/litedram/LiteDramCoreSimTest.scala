// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory.litedram

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._

/**
 * Stage-A silicon-debug simulation of the LiteDRAM core (Verilator).
 *
 * Runs the generated `litedram_core.v` + Gowin `prim_sim.v` behavioral
 * models with the DDR pads left floating (no memory model yet): the JEDEC
 * init sequence is fire-and-forget, so `init_done` must assert from the
 * CRG/sequencer alone. Catches reset/clock-structure bugs (cf. the
 * self-holding sync-reset deadlock and the LiteX-PLL static clocks that
 * muted the board) without any silicon round-trip.
 *
 * Verilator adaptations (test-time, nothing committed):
 * - `prim_sim.v` is copied to `dram/out/sim/` with `module rPLL` stripped:
 *   the vendor model is `#delay`-driven (combinational loop once Verilator
 *   ignores delays). A functional stub (ideal 54/27MHz + LOCK) stands in.
 * - prim_sim references the `GSR` scope with no instance: `GSR.GSRO`
 *   is constant-folded to `1'b1` (GSRI passthrough at 1 = global reset
 *   inactive; our POR handles reset, Verilator honors init values).
 *
 * Run: `test-all -k LiteDramCoreSim --retry 0` (Windows; needs the Gowin
 * IDE simlib, see GOWIN_SIMLIB_DIR). Wave: add `.withWave` to SimConfig.
 */
object GowinSimPrep {
  // Functional rPLL stub for Verilator (ideal clocks, LOCK after 256 ref
  // cycles; phase relationship sys/sys2x preserved for the 1:2 DFI).
  val rpllStub: String =
    """module rPLL (
      |  output reg CLKOUT, output reg CLKOUTP, output reg CLKOUTD, output reg CLKOUTD3,
      |  output reg LOCK,
      |  input CLKIN, input CLKFB,
      |  input [5:0] FBDSEL, input [5:0] IDSEL, input [5:0] ODSEL,
      |  input [3:0] PSDA, input [3:0] DUTYDA, input [3:0] FDLY,
      |  input RESET, input RESET_P
      |);
      |  parameter CLKFB_SEL="internal", CLKOUTD3_SRC="CLKOUT", CLKOUTD_BYPASS="false",
      |    CLKOUTD_SRC="CLKOUT", CLKOUTP_BYPASS="false", CLKOUT_BYPASS="false",
      |    CLKOUT_DLY_STEP=0, CLKOUT_FT_DIR=1, CLKOUTP_DLY_STEP=0, CLKOUTP_FT_DIR=1,
      |    DEVICE="GW2A-18C", DUTYDA_SEL="1000", DYN_DA_EN="false", DYN_FBDIV_SEL="false",
      |    DYN_IDIV_SEL="false", DYN_ODIV_SEL="false", DYN_SDIV_SEL=2, FBDIV_SEL=1,
      |    FCLKIN="27", IDIV_SEL=0, ODIV_SEL=16, PSDA_SEL="0100";
      |  integer lockCnt;
      |  initial begin
      |    CLKOUT = 0; CLKOUTP = 0; CLKOUTD = 0; CLKOUTD3 = 0; LOCK = 0; lockCnt = 0;
      |  end
      |  always @(posedge CLKIN or posedge RESET) begin
      |    if (RESET) begin
      |      CLKOUTD <= 0; LOCK <= 0; lockCnt <= 0;
      |    end else begin
      |      CLKOUTD <= ~CLKOUTD;
      |      if (lockCnt < 1000000) lockCnt <= lockCnt + 1;
      |      if (lockCnt > 255) LOCK <= 1;
      |    end
      |  end
      |  always @(posedge CLKIN or negedge CLKIN or posedge RESET) begin
      |    if (RESET) CLKOUT <= 0;
      |    else CLKOUT <= ~CLKOUT;
      |  end
      |endmodule
      |""".stripMargin

  /** Returns (primNoRpll, rpllStub) absolute paths, generating if needed. */
  def prepare(): (String, String) = {    val ideDir = sys.env.getOrElse("GOWIN_SIMLIB_DIR",
      "C:/Gowin/Gowin_V1.9.11.03_Education_x64/IDE/simlib/gw2a")
    val src = new java.io.File(ideDir, "prim_sim.v")
    assert(src.isFile, s"Gowin prim_sim.v not found in $ideDir (GOWIN_SIMLIB_DIR?)")
    val outDir = new java.io.File(new java.io.File(LiteDramCore.corePath).getParent, "sim")
    outDir.mkdirs()
    val noPll = new java.io.File(outDir, "prim_sim_norpll.v")
    val stub = new java.io.File(outDir, "rpll_stub.v")
    if (!noPll.isFile) {
      val lines = scala.io.Source.fromFile(src).getLines().toSeq
      val kept = new StringBuilder
      var skip = false
      for (ln <- lines) {
        if (ln.startsWith("module rPLL ")) skip = true
        if (!skip) {
          // GSR is never instantiated: its only use is `GSR.GSRO` (= GSRI
          // passthrough). With GSRI=1 the global reset is inactive, i.e.
          // grstn is 1 either way -> constant-fold the reference. (Our POR
          // handles reset; Verilator honors init values.)
          kept.append(ln.replace("GSR.GSRO", "1'b1")).append("\n")
        }
        if (skip && ln.startsWith("endmodule")) skip = false
      }
      val w = new java.io.PrintWriter(noPll)
      try w.write(kept.toString()) finally w.close()
    }
    if (!stub.isFile) {
      val w = new java.io.PrintWriter(stub)
      try w.write(rpllStub) finally w.close()
    }
    (noPll.getAbsolutePath, stub.getAbsolutePath)
  }

  // Single-x16 Micron wrapper (dqs_n = ~dqs_p; TDQS unused in x16).
  // Our core has no dqs_n port (SSTL15D tile generates N); in sim the
  // complement is exact (model drives both on reads, we drive P on writes).
  val ddrWrap: String =
    """module ddr3_wrap (
      |  input rst_n, input ck, input ck_n, input cke, input cs_n,
      |  input ras_n, input cas_n, input we_n,
      |  inout [1:0] dm, input [2:0] ba, input [12:0] addr,
      |  inout [15:0] dq, inout [1:0] dqs_p, input odt
      |);
      |  wire [1:0] dqs_n;
      |  assign dqs_n = ~dqs_p;
      |  // Relax model-side checks for bring-up (observe functionally first;
      |  // re-enable to audit MR legality once green). NOTE: mrbits stays 1:
      |  // DLL-off only warns ("not fully modeled"), illegal MR bits error.
      |  // STOP_ON_ERROR=0: OBSERVE mode (violations print, never $stop;
      |  // DLL-off init tracking can never complete by construction).
      |  defparam mem.check_strict_timing = 0;
      |  defparam mem.check_strict_mrbits = 1;
      |  defparam mem.STOP_ON_ERROR = 0;
      |  ddr3 mem (
      |    .rst_n(rst_n), .ck(ck), .ck_n(ck_n), .cke(cke), .cs_n(cs_n),
      |    .ras_n(ras_n), .cas_n(cas_n), .we_n(we_n), .dm_tdqs(dm), .ba(ba),
      |    .addr(addr), .dq(dq), .dqs(dqs_p), .dqs_n(dqs_n), .odt(odt)
      |  );
      |endmodule
      |""".stripMargin

  /** Hierarchical $display monitor, bound into SimTopB (sibling scope).
   * Verilator cannot resolve absolute TOP.* paths from a plain module, but
   * `bind` places the probe inside SimTopB where `core.*` resolves.
   * v4: which bankmachine gets the request + its cmd (25 bits):
   * {init, npcv, cc.v, cc.r, reqv[7:0], bmc[7:0], cs, ras, cas, we, bvalid} */
  val liteDbg: String =
    """bind SimTopB lite_dbg_probe probe_i();
      |module lite_dbg_probe;
      |  reg [24:0] prev;
      |  wire [24:0] cur;
      |  assign cur = {core.init_done,
      |    core.new_port_cmd_valid,
      |    core.core_cmd_valid, core.core_cmd_ready,
      |    core.core_bankmachine7_req_valid, core.core_bankmachine6_req_valid,
      |    core.core_bankmachine5_req_valid, core.core_bankmachine4_req_valid,
      |    core.core_bankmachine3_req_valid, core.core_bankmachine2_req_valid,
      |    core.core_bankmachine1_req_valid, core.core_bankmachine0_req_valid,
      |    core.core_bankmachine7_cmd_valid, core.core_bankmachine6_cmd_valid,
      |    core.core_bankmachine5_cmd_valid, core.core_bankmachine4_cmd_valid,
      |    core.core_bankmachine3_cmd_valid, core.core_bankmachine2_cmd_valid,
      |    core.core_bankmachine1_cmd_valid, core.core_bankmachine0_cmd_valid,
      |    core.gw2ddrphy_dfi_p0_cs_n, core.gw2ddrphy_dfi_p0_ras_n,
      |    core.gw2ddrphy_dfi_p0_cas_n, core.gw2ddrphy_dfi_p0_we_n,
      |    core.b_valid};
      |  initial prev = 25'd0;
      |  always @(posedge core.sys_clk) begin
      |    if (cur != prev)
      |      $display("LITEDBG t=%0t %b", $time, cur);
      |    prev <= cur;
      |  end
      |endmodule
      |""".stripMargin

  def stageDbgMon(simDir: String): String = {
    val f = new java.io.File(simDir, "lite_dbg.v")
    if (!f.isFile) {
      val w = new java.io.PrintWriter(f)
      try w.write(liteDbg) finally w.close()
    }
    f.getAbsolutePath
  }
  def stageDdr(): String = {
    val simDir = new java.io.File(new java.io.File(LiteDramCore.corePath).getParent, "sim")
    simDir.mkdirs()
    val refs = sys.env.getOrElse("DDR3_MODEL_DIR", "E:/refs")
    val pairs = Seq(
      (new java.io.File(refs, "ddr3-buttercutter/ddr3.v"), "ddr3.v"),
      (new java.io.File(refs, "ddr3-micron-model/1024Mb_ddr3_parameters.vh"),
        "1024Mb_ddr3_parameters.vh")
    )
    for ((src, name) <- pairs) {
      val dst = new java.io.File(simDir, name)
      if (!dst.isFile) {
        assert(src.isFile, s"DDR3 model file missing: ${src.getPath} (DDR3_MODEL_DIR?)")
        java.nio.file.Files.copy(src.toPath, dst.toPath)
      }
    }
    // Array-backed model, not file-backed: buttercutter's ddr3.v hardcodes
    // `define MAX_MEM (per-bank /tmp files via $fseek/$fscanf, which
    // Verilator cannot do sparsely -> $finish on first read). Dropping the
    // define selects the plain `memory[]` array (also drops the
    // $value$plusargs block with it).
    val stagedModel = new java.io.File(simDir, "ddr3.v")
    var text = scala.io.Source.fromFile(stagedModel).mkString
    if (text.contains("`define MAX_MEM")) {
      text = text.replace("`define MAX_MEM", "// `define MAX_MEM (disabled for Verilator: array model)")
      val w = new java.io.PrintWriter(stagedModel)
      try w.write(text) finally w.close()
    }
    val wrap = new java.io.File(simDir, "ddr3_wrap.v")
    if (!wrap.isFile) {
      val w = new java.io.PrintWriter(wrap)
      try w.write(ddrWrap) finally w.close()
    }
    simDir.getAbsolutePath
  }
}

class LiteDramCoreSimTest extends AnyFunSuite {

  /** Test wrapper: BlackBox core, AXI idle, DDR pads floating. */
  class SimTop extends Component {
    val io = new Bundle {
      val resetN = in(Bool())
      val initDone = out(Bool())
      val pllLocked = out(Bool())
      val dbgStep = out(Bits(8 bits))
      val dbgRst = out(Bits(8 bits))
    }
    val core = new LiteDramCore()
    val (primNoRpll, rpllStub) = GowinSimPrep.prepare()
    core.addRTLPath(primNoRpll)
    core.addRTLPath(rpllStub)
    core.mapClockDomain(clock = core.io.clk27)
    core.io.reset_n := io.resetN
    io.initDone := core.io.init_done
    io.pllLocked := core.io.pll_locked
    io.dbgStep := core.io.dbg_step
    io.dbgRst := core.io.dbg_rst

    // AXI (LiteX streaming flavour): idle, ready to accept.
    core.io.aw_valid := False
    core.io.aw_first := True
    core.io.aw_last := True
    core.io.aw_payload_addr := 0
    core.io.aw_payload_burst := 0
    core.io.aw_payload_len := 0
    core.io.aw_payload_size := 0
    core.io.aw_payload_lock := False
    core.io.aw_payload_prot := 0
    core.io.aw_payload_cache := 0
    core.io.aw_payload_qos := 0
    core.io.aw_payload_region := 0
    core.io.aw_param_id := 0
    core.io.aw_param_dest := False
    core.io.aw_param_user := False
    core.io.w_valid := False
    core.io.w_first := True
    core.io.w_last := True
    core.io.w_payload_data := 0
    core.io.w_payload_strb := 0
    core.io.w_param_id := 0
    core.io.w_param_dest := False
    core.io.w_param_user := False
    core.io.b_ready := True
    core.io.ar_valid := False
    core.io.ar_first := True
    core.io.ar_last := True
    core.io.ar_payload_addr := 0
    core.io.ar_payload_burst := 0
    core.io.ar_payload_len := 0
    core.io.ar_payload_size := 0
    core.io.ar_payload_lock := False
    core.io.ar_payload_prot := 0
    core.io.ar_payload_cache := 0
    core.io.ar_payload_qos := 0
    core.io.ar_payload_region := 0
    core.io.ar_param_id := 0
    core.io.ar_param_dest := False
    core.io.ar_param_user := False
    core.io.r_ready := True

    // DDR pads float (stage A: no memory model; outputs dangle).
    val dq = Analog(Bits(16 bits))
    val dqs = Analog(Bits(2 bits))
    core.io.dq := dq
    core.io.dqs_p := dqs
    val a = Bits(14 bits)
    val ba = Bits(3 bits)
    val dm = Bits(2 bits)
    a := core.io.a
    ba := core.io.ba
    dm := core.io.dm
    val rasN, casN, weN, csN, clkP, clkN, cke, odt, rstN = Bool()
    rasN := core.io.ras_n
    casN := core.io.cas_n
    weN := core.io.we_n
    csN := core.io.cs_n
    clkP := core.io.clk_p
    clkN := core.io.clk_n
    cke := core.io.cke
    odt := core.io.odt
    rstN := core.io.reset_n_1
    val awReady, wReady, bValid, bFirst, bLast, arReady = Bool()
    awReady := core.io.aw_ready
    wReady := core.io.w_ready
    bValid := core.io.b_valid
    bFirst := core.io.b_first
    bLast := core.io.b_last
    arReady := core.io.ar_ready
    val bResp = Bits(2 bits)
    bResp := core.io.b_payload_resp
    val bId = UInt(4 bits)
    bId := core.io.b_param_id
    val bDest, bUser = Bool()
    bDest := core.io.b_param_dest
    bUser := core.io.b_param_user
    val rValid, rFirst, rLast = Bool()
    rValid := core.io.r_valid
    rFirst := core.io.r_first
    rLast := core.io.r_last
    val rResp = Bits(2 bits)
    rResp := core.io.r_payload_resp
    val rData = Bits(64 bits)
    rData := core.io.r_payload_data
    val rId = UInt(4 bits)
    rId := core.io.r_param_id
    val rDest, rUser = Bool()
    rDest := core.io.r_param_dest
    rUser := core.io.r_param_user
  }

  test("litedram-core-init (verilator, stage A: init_done, pads floating)") {
    // --no-timing: prim_sim transport delays (#0.025/#0.2, fidelity only)
    // must be ignored (rPLL oscillator is stubbed, see GowinSimPrep).
    // -Wno-*: vendor models are warning-noisy (COMBDLY from ignored
    // delays, latch inside the DLL model, unconnected stub pins);
    // all reviewed benign for functional init validation.
    SimConfig
      .addSimulatorFlag("--no-timing")
      .addSimulatorFlag("-Wno-COMBDLY")
      .addSimulatorFlag("-Wno-IEEEMAYDEPRECATE")
      .addSimulatorFlag("-Wno-REALCVT")
      .addSimulatorFlag("-Wno-IMPLICIT")
      .addSimulatorFlag("-Wno-PINMISSING")
      .addSimulatorFlag("-Wno-CASEINCOMPLETE")
      .addSimulatorFlag("-Wno-MISINDENT")
      .addSimulatorFlag("-Wno-LATCH")
      .compile(new SimTop).doSim { dut =>
      // 27 MHz board clock.
      dut.clockDomain.forkStimulus(period = 37037)

      dut.io.resetN #= false
      dut.clockDomain.waitSampling(10)
      dut.io.resetN #= true

      // rPLL lock first (~ms in prim_sim time).
      var waited = 0
      while (!dut.io.pllLocked.toBoolean && waited < 100000) {
        dut.clockDomain.waitSampling(1000)
        waited += 1000
      }
      assert(dut.io.pllLocked.toBoolean, "rPLL never locked in sim (prim_sim model?)")

      // JEDEC init: POR 65536 + ~5ms script ≈ 220k cycles; watchdog 500k.
      waited = 0
      while (!dut.io.initDone.toBoolean && waited < 500000) {
        dut.clockDomain.waitSampling(1000)
        waited += 1000
      }
      val step = dut.io.dbgStep.toBigInt
      val rst = dut.io.dbgRst.toBigInt
      assert(dut.io.initDone.toBoolean,
        f"init_done stuck 0 after ${waited}cyc (dbg_step=0x$step%02X dbg_rst=0x$rst%02X)")
    }
  }

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
        cd.waitSamplingWhere(dut.io.awReady.toBoolean && dut.io.wReady.toBoolean)
        dut.io.awValid #= false
        dut.io.wValid #= false
        var n = 0
        while (!dut.io.bValid.toBoolean && n < 20000) { cd.waitSampling(10); n += 10 }
        assert(dut.io.bValid.toBoolean, f"write b_valid timeout @word $wordAddr%x")
        assert(dut.io.bResp.toBigInt == 0, "write RESP not OKAY")
        cd.waitSampling(5)
      }

      def axiRead(wordAddr: BigInt): BigInt = {
        dut.io.arAddr #= wordAddr
        dut.io.arValid #= true
        cd.waitSamplingWhere(dut.io.arReady.toBoolean)
        dut.io.arValid #= false
        var n = 0
        while (!dut.io.rValid.toBoolean && n < 20000) { cd.waitSampling(10); n += 10 }
        assert(dut.io.rValid.toBoolean, f"read r_valid timeout @word $wordAddr%x (PHY read hang?)")
        val d = dut.io.rData.toBigInt
        cd.waitSampling(5)
        d
      }

      // Word addresses span columns/rows (byte addr = word << 3).
      val vectors = Seq(
        (BigInt(0x0), BigInt("0123456789ABCDEF", 16)),
        (BigInt(0x1), BigInt("FEDCBA9876543210", 16)),
        (BigInt(0x100), BigInt("A5A5A5A55A5A5A5A", 16)),
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
