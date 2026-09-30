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
  // cycles; sys/sys2x 1:2 ratio exact, phase-locked).
  // IMPORTANT: this stub expects 54 MHz on CLKIN (the TB drives io.clk27
  // at 54 MHz via forkStimulus(18519), NOT the board 27 MHz). Division is
  // exact: sys2x = CLKIN toggled on both edges (54 MHz), sys = CLKIN
  // toggled on rising edges only (27 MHz). A single-rate toggle on both
  // edges yields F/1, NOT 2F (each input period holds 2 toggles = 1 output
  // period) -- a previous version ran sys2x at 27 MHz (1:1), silently
  // halving every DFI/DQS rate (measured: 37 ns probe spacing). The real
  // silicon rPLL (27 MHz board clock, wizard dividers) is unaffected.
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
      |      LOCK <= 0; lockCnt <= 0;
      |    end else begin
      |      if (lockCnt < 1000000) lockCnt <= lockCnt + 1;
      |      if (lockCnt > 255) LOCK <= 1;
      |    end
      |  end
      |  // sys2x = 54 MHz (CLKIN at 54 MHz toggled both edges = 54 MHz).
      |  always @(posedge CLKIN or negedge CLKIN or posedge RESET) begin
      |    if (RESET) CLKOUT <= 0;
      |    else CLKOUT <= ~CLKOUT;
      |  end
      |  // sys = 27 MHz (rising edges only = divide by 2).
      |  always @(posedge CLKIN or posedge RESET) begin
      |    if (RESET) CLKOUTD <= 0;
      |    else CLKOUTD <= ~CLKOUTD;
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
    // Always rewrite (see lite_dbg.v above): the stub layout evolves and
    // a stale cache would silently run the wrong clocks.
    val wstub = new java.io.PrintWriter(stub)
    try wstub.write(rpllStub) finally wstub.close()
    (noPll.getAbsolutePath, stub.getAbsolutePath)
  }

  // Single-x16 Micron wrapper (dqs_n = ~dqs_p; TDQS unused in x16).
  // Our core has no dqs_n port (SSTL15D tile generates N); in sim the
  // complement is exact (model drives both on reads, we drive P on writes).
  // TEST-ONLY MODEL HACK: force the Micron init tracker complete. With
  // MR1 DLL-off the model can never see a legal init (its init_step gate)
  // so it would drop ALL state updates (ACT never opens, WR never lands
  // in memory[], RD returns zero-init) while still checking commands.
  // Forcing init_done=1 keeps every legality/timing check active and only
  // enables the functional array behavior. Revisit (DLL-on core) if the
  // DLL-off DQS read path misbehaves.
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
      |  initial force mem.init_done = 1'b1;
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
   * v12: v11 signals, but sampled EVERY sys2x edge (54 MHz: no sampling
   * blindness for 1-cycle pulses/glitches) inside ONE bounded window
   * (init_done + 15000 edges ~= 278us: burst + reads). ~1 MB log.
   * Layout (51 bits):
   * {init, aw_v, aw_r, wsrc_v, lvl[4:0], can_wr,
   *  npc_v, npc_r, npc_last, npc_we, pc_v, pc_r,
   *  conv_st[1:0], rr0, reqv0, awrdy, wdatav, wdata[15:0],
   *  bm_st[2:0], bm_cv, bm_cr, bm_rr,
   *  cs, ras, cas, we, bvalid, npw_v, npw_r} */
  val liteDbg: String =
    """bind SimTopB lite_dbg_probe probe_i();
      |bind SimTopD lite_dbg_probe probe_d();
      |module lite_dbg_probe;
      |  wire [50:0] cur;
      |  assign cur = {core.init_done,
      |    core.write_aw_valid, core.write_aw_ready,
      |    core.write_w_buffer_source_valid,
      |    core.write_w_buffer_level1,
      |    core.write_can_write,
      |    core.new_port_cmd_valid, core.new_port_cmd_ready,
      |    core.new_port_cmd_last, core.new_port_cmd_payload_we,
      |    core.port_cmd_valid, core.port_cmd_ready,
      |    core.litedramcore_state,
      |    core.litedramcore_roundrobin0_request,
      |    core.core_bankmachine0_req_valid,
      |    core.write_aw_ready,
      |    core.litedramnativeportconverter_wdata_buffer_source_source_valid,
      |    core.litedramnativeportconverter_wdata_buffer_source_source_payload_data[15:0],
      |    core.litedramcore_bankmachine0_state,
      |    core.core_bankmachine0_cmd_valid, core.core_bankmachine0_cmd_ready,
      |    core.core_bankmachine0_refresh_req,
      |    core.gw2ddrphy_dfi_p0_cs_n, core.gw2ddrphy_dfi_p0_ras_n,
      |    core.gw2ddrphy_dfi_p0_cas_n, core.gw2ddrphy_dfi_p0_we_n,
      |    core.b_valid,
      |    core.new_port_wdata_valid, core.new_port_wdata_ready};
      |  reg en;
      |  reg done;
      |  reg [14:0] cnt;
      |  initial begin en = 1'b0; done = 1'b0; cnt = 15'd0; end
      |  always @(posedge core.sys2x_clk) begin
      |    if (!en && !done && core.init_done) begin en <= 1'b1; cnt <= 15'd0; end
      |    else if (en) begin
      |      $display("LITEDBG t=%0t %b", $time, cur);
      |      cnt <= cnt + 15'd1;
      |      if (cnt >= 15'd15000) begin en <= 1'b0; done <= 1'b1; end
      |    end
      |  end
      |endmodule
      |""".stripMargin

  def stageDbgMon(simDir: String): String = {
    val f = new java.io.File(simDir, "lite_dbg.v")
    // Always rewrite: the probe layout evolves (v3/v4/v5...), a stale
    // cached file would silently run the wrong monitor.
    val w = new java.io.PrintWriter(f)
    try w.write(liteDbg) finally w.close()
    f.getAbsolutePath
  }

  // DQS read-capture monitor (stage D), v2: ASYNC change-detect on the
  // Gowin DQS primitive internals (exact $time per transition — no
  // sampling alias): RPOINT + WPOINT + RVALID + READ per lane, plus the
  // DFI rddata_valid strobe. Gives the EXACT R timeline (start lag,
  // dwell, rate, orbit, rest) needed to place the sample tap.
  // Bounded: max 3000 lines.
  val dqsMon: String =
    """bind SimTopD dqs_mon_probe probe_dqs();
      |module dqs_mon_probe;
      |  reg [11:0] n;
      |  reg done;
      |  initial begin n = 12'd0; done = 1'b0; end
      |  task automatic emit;
      |    begin
      |      if (core.init_done && !done) begin
      |        $display("DQSMON t=%0t re=%b wpt0=%d rpt0=%d rv0=%b wpt1=%d rpt1=%d rv1=%b rdv=%b",
      |          $time, core.gw2ddrphy_dqs_re,
      |          core.DQS.WPOINT, core.DQS.RPOINT, core.DQS.RVALID,
      |          core.DQS_1.WPOINT, core.DQS_1.RPOINT, core.DQS_1.RVALID,
      |          core.gw2ddrphy_dfi_p0_rddata_valid);
      |        n <= n + 12'd1;
      |        if (n >= 12'd3000) done <= 1'b1;
      |      end
      |    end
      |  endtask
      |  always @(core.DQS.WPOINT) emit();
      |  always @(core.DQS.RPOINT) emit();
      |  always @(core.DQS.RVALID) emit();
      |  always @(core.DQS_1.WPOINT) emit();
      |  always @(core.DQS_1.RPOINT) emit();
      |  always @(core.DQS_1.RVALID) emit();
      |  always @(core.gw2ddrphy_dqs_re) emit();
      |  always @(core.gw2ddrphy_dfi_p0_rddata_valid) emit();
      |endmodule
      |""".stripMargin

  def stageDqsMon(simDir: String): String = {
    val f = new java.io.File(simDir, "dqs_mon.v")
    // Always rewrite (same reason as stageDbgMon).
    val w = new java.io.PrintWriter(f)
    try w.write(dqsMon) finally w.close()
    f.getAbsolutePath
  }

  // DFI read-word monitor (stage D): prints the DFI rddata phases
  // (64 bits each; this halfrate PHY has p0/p1 only) + rddata_valid
  // strobes on ANY change after init_done (bounded: 3000 lines). Shows
  // exactly what the PHY presents to the controller each cycle during
  // the RPOINT transit + settle, killing all speculation about
  // pipeline depth vs sample time.
  val dfiMon: String =
    """bind SimTopD dfi_mon_probe probe_dfi();
      |module dfi_mon_probe;
      |  wire [63:0] p0 = core.gw2ddrphy_dfi_p0_rddata;
      |  wire [63:0] p1 = core.gw2ddrphy_dfi_p1_rddata;
      |  wire v0 = core.gw2ddrphy_dfi_p0_rddata_valid;
      |  wire v1 = core.gw2ddrphy_dfi_p1_rddata_valid;
      |  reg [129:0] prev;
      |  reg [11:0] n;
      |  reg init;
      |  reg done;
      |  wire [129:0] cur;
      |  assign cur = {p0, p1, v0, v1};
      |  initial begin prev = 130'd0; n = 12'd0; init = 1'b0; done = 1'b0; end
      |  always @(negedge core.sys2x_clk) begin
      |    if (core.init_done && !done) begin
      |      if (!init) begin init <= 1'b1; prev <= cur; end
      |      else if (cur != prev) begin
      |        $display("DFIMON t=%0t p0=%h p1=%h v=%b%b", $time, p0, p1, v0, v1);
      |        n <= n + 12'd1;
      |        prev <= cur;
      |        if (n >= 12'd3000) done <= 1'b1;
      |      end
      |    end
      |  end
      |endmodule
      |""".stripMargin

  def stageDfiMon(simDir: String): String = {
    val f = new java.io.File(simDir, "dfi_mon.v")
    // Always rewrite (same reason as stageDbgMon).
    val w = new java.io.PrintWriter(f)
    try w.write(dfiMon) finally w.close()
    f.getAbsolutePath
  }

  // SIM-ONLY core copy with the PHY read path re-phased for the
  // zero-delay prim_sim models (see DQSMON findings).
  // - dqs_re (DQS primitive READ = capture window): LiteX rdtap
  //   (delayline2|3) is tuned for silicon DLL/IO delays; in sim the
  //   window opens ~1 beat late (beat0 missed). Moving ONLY dqs_re does
  //   not alter the controller contract (same delay line, same depth).
  // - rddata_valid tap (DFI sample strobe): stock = delayline11. The
  //   RPOINT transit (0>1>3>2>6, ~+3.5..+5 sys after rddata_en) is over
  //   ~6 sys before the stock sample, so the settled pipeline shows a
  //   single 2-beat slot broadcast ([x,y,x,y]). Sampling earlier walks
  //   the sample window back across the transit; at the right tap the
  //   4-FCLK window covers 4 filled slots = full 8-beat word.
  // The committed dram/out/litedram_core.v is NEVER modified (silicon
  // flow untouched); the patched copy is rewritten per run into simDir.
  def stageCoreSim(simDir: String, dqsTaps: Seq[Int], rdvTap: Int): String = {
    val src = scala.io.Source.fromFile(LiteDramCore.corePath)
    val text = try src.mkString finally src.close()
    val dqsFrom = "assign gw2ddrphy_dqs_re = " +
      "(gw2ddrphy_rddata_en_tappeddelayline2 | gw2ddrphy_rddata_en_tappeddelayline3);"
    assert(text.contains(dqsFrom),
      "dqs_re pattern not found in generated core (regenerate via dram-gen?)")
    val dqsExpr = dqsTaps.map(t => s"gw2ddrphy_rddata_en_tappeddelayline$t").mkString(" | ")
    var patched = text.replace(dqsFrom,
      s"assign gw2ddrphy_dqs_re = ($dqsExpr); // SIM-ONLY re-phase (stageCoreSim)")
    val rdvFrom0 = "assign gw2ddrphy_dfi_p0_rddata_valid = gw2ddrphy_rddata_en_tappeddelayline11;"
    val rdvFrom1 = "assign gw2ddrphy_dfi_p1_rddata_valid = gw2ddrphy_rddata_en_tappeddelayline11;"
    assert(patched.contains(rdvFrom0) && patched.contains(rdvFrom1),
      "rddata_valid pattern not found in generated core")
    patched = patched
      .replace(rdvFrom0,
        s"assign gw2ddrphy_dfi_p0_rddata_valid = gw2ddrphy_rddata_en_tappeddelayline$rdvTap; // SIM-ONLY (stageCoreSim)")
      .replace(rdvFrom1,
        s"assign gw2ddrphy_dfi_p1_rddata_valid = gw2ddrphy_rddata_en_tappeddelayline$rdvTap; // SIM-ONLY (stageCoreSim)")
    val f = new java.io.File(simDir, "litedram_core_sim.v")
    val w = new java.io.PrintWriter(f)
    try w.write(patched) finally w.close()
    f.getAbsolutePath
  }

  // Homegrown DFI responder (replaces the Micron model for fast debug).
  // Drop-in on the DDR pads: decodes ACT/WR/RD on rising ck, captures WR
  // data framed on dqs_p[0] edges into a 16-entry CAM, replays on RD with
  // a tunable read latency. Prints ONLY our traffic (no model storm).
  // Margin char is '#' (Verilog '|' would break stripMargin).
  // Assumptions (adjust once, empirically): word = {samp7..samp0} with
  // samp0 = first beat = DFI wdata[15:0]; SKIP_PRE skips the WR preamble
  // edge; RL = read latency in ck.
  val dfiResp: String =
    """module dfi_resp (
      #  input rst_n, input ck, input ck_n, input cke, input cs_n,
      #  input ras_n, input cas_n, input we_n,
      #  input [1:0] dm, input [2:0] ba, input [12:0] addr,
      #  inout [15:0] dq, inout [1:0] dqs_p, input odt
      #);
      #  parameter RL = 6;
      #  parameter WSKIP = 2;
      #  // Tag masks A10 (auto-precharge flag, not column).
      #  wire [25:0] ctag = {ba, addr[12:11], 1'b0, addr[9:0]};
      #  // Single-sample command decode. (A 2-sample qualifier was tried:
      #  // it eats real single-cycle DFI commands while passing 2-ck-wide
      #  // transitional aliases, exactly inverted. Transitional phantoms
      #  // are rare and identifiable by timestamp vs the TB windows.)
      #  reg wr_v, rd_v;
      #  reg [25:0] c_tag;
      #  always @(posedge ck) begin
      #    if (~rst_n) begin wr_v <= 1'b0; rd_v <= 1'b0; end
      #    else begin
      #      wr_v <= ~cs_n & ras_n & ~cas_n & ~we_n;
      #      rd_v <= ~cs_n & ras_n & ~cas_n & we_n;
      #    end
      #    c_tag <= ctag;
      #  end
      #  // 16-entry CAM tag -> 128-bit word.
      #  reg [25:0] cam_tag [0:15];
      #  reg [127:0] cam_dat [0:15];
      #  reg [15:0] cam_vld;
      #  reg [3:0] cam_ptr;
      #  integer i;
      #  initial begin cam_vld = 16'd0; cam_ptr = 4'd0; end
      #  // Write capture window (ck domain) + epoch for edge-domain reset.
      #  reg [3:0] warm;
      #  reg [3:0] warepoch;
      #  reg [25:0] w_tag;
      #  always @(posedge ck) begin
      #    if (~rst_n) begin warm <= 4'd0; end
      #    else if (wr_v) begin warm <= 4'd12; w_tag <= c_tag; warepoch <= warepoch + 4'd1; end
      #    else if (warm != 4'd0) warm <= warm - 4'd1;
      #  end
      #  initial warepoch = 4'd0;
      #  wire wactive = (warm != 4'd0);
      #  // Edge-domain capture (single driver each): epoch tags the burst.
      #  // CALIBRATION: keep 12 edges, print all (alignment chosen offline).
      #  reg [3:0] wcnt;
      #  reg [3:0] wepoch;
      #  reg [15:0] wsamp [0:11];
      #  initial begin wcnt = 4'd0; wepoch = 4'd0; end
      #  always @(posedge dqs_p[0] or negedge dqs_p[0]) begin
      #    if (wactive && (wepoch != warepoch)) begin wepoch <= warepoch; wcnt <= 4'd0; end
      #    else if (wactive && (wcnt < 4'd12)) begin wsamp[wcnt] <= dq; wcnt <= wcnt + 4'd1; end
      #  end
      #  // Commit at window close (ck domain).
      #  reg [3:0] warm_d;
      #  // Calibrated 2026-09-29: DQS burst = 2 idle edges (1-CK preamble)
      #  // + 8 data + idle. Word = samples [9:2] (proven: OUR bytes land
      #  // there, CDEF..FEDC in order).
      #  wire [127:0] wdata = {wsamp[9], wsamp[8], wsamp[7], wsamp[6], wsamp[5], wsamp[4], wsamp[3], wsamp[2]};
      #  always @(posedge ck) begin
      #    warm_d <= warm;
      #    if ((warm_d != 4'd0) && (warm == 4'd0)) begin
      #      cam_tag[cam_ptr] <= w_tag;
      #      cam_dat[cam_ptr] <= wdata;
      #      cam_vld[cam_ptr] <= 1'b1;
      #      $display("DFIRESP t=%0t WR tag=%h beats=%0d dm=%b data=%h raw=%h_%h_%h_%h_%h_%h_%h_%h_%h_%h_%h_%h", $time, w_tag, wcnt, dm, wdata,
      #        wsamp[11], wsamp[10], wsamp[9], wsamp[8], wsamp[7], wsamp[6], wsamp[5], wsamp[4], wsamp[3], wsamp[2], wsamp[1], wsamp[0]);
      #      cam_ptr <= cam_ptr + 4'd1;
      #    end
      #  end
      #  // Read replay: lookup on RD, drive preamble + 8 beats + postamble.
      #  reg rd_on;
      #  reg [3:0] rdcnt;
      #  reg [127:0] rword;
      #  reg rhit;
      #  reg [25:0] r_tag;
      #  always @(posedge ck) begin
      #    if (~rst_n) rd_on <= 1'b0;
      #    else if (rd_v) begin
      #      rd_on <= 1'b1; rdcnt <= 4'd0; rhit <= 1'b0; rword <= 128'd0; r_tag <= ctag;
      #      for (i = 0; i < 16; i = i + 1)
      #        if (cam_vld[i] && (cam_tag[i] == ctag)) begin rhit <= 1'b1; rword <= cam_dat[i]; end
      #    end
      #    else if (rd_on) begin
      #      rdcnt <= rdcnt + 4'd1;
      #      if (rdcnt == 4'd10) rd_on <= 1'b0;
      #    end
      #  end
      #  wire rd_oe = rd_on && (rdcnt >= RL-2) && (rdcnt <= RL+3);
      #  reg ph;
      #  reg [3:0] bcnt;
      #  reg [1:0] dqs_o;
      #  reg [15:0] dq_o;
      #  initial begin ph = 1'b0; bcnt = 4'd0; dqs_o = 2'b00; dq_o = 16'd0; end
      #  always @(posedge ck or negedge ck) begin
      #    ph <= ~ph;
      #    // Pre-drive beat0 (+DQS low) on RD: IDES samples PRE-edge values,
      #    // so beat[k] must be stable BEFORE DQS edge k (driving it AT the
      #    // edge samples stale). Fires on both rd_v edges (idempotent).
      #    if (rd_v) begin bcnt <= 4'd0; dq_o <= rword[15:0]; dqs_o <= 2'b00; end
      #    else if (rd_oe && (rdcnt >= RL-1) && (rdcnt <= RL+2)) begin
      #      dqs_o <= {2{ph}};
      #      if (bcnt < 4'd7) dq_o <= rhit ? rword[(bcnt+1)*16 +: 16] : 16'hBEEF;
      #      bcnt <= bcnt + 4'd1;
      #    end
      #    else if (rd_oe) begin dqs_o <= 2'b00; dq_o <= 16'd0; end
      #  end
      #  // Drive ONLY on HIT: a miss (phantom RD on idle bus, or read of
      #  // never-written addr) must stay high-Z, otherwise its DQS/DQ
      #  // window contends with a real WR burst nearby and corrupts both.
      #  // (Real DRAM always drives on RD; for debug, silence > BEEF.)
      #  assign dqs_p = (rd_oe & rhit) ? dqs_o : 2'bzz;
      #  assign dq = (rd_oe & rhit) ? dq_o : 16'hzzzz;
      #  // Read summary (hit resolved a cycle after RD).
      #  reg rd_vd;
      #  always @(posedge ck) begin
      #    rd_vd <= rd_v;
      #    if (rd_vd) $display("DFIRESP t=%0t RD tag=%h hit=%b data=%h", $time, r_tag, rhit, rword);
      #  end
      #endmodule
      #""".stripMargin('#')

  def stageResp(simDir: String): String = {
    val f = new java.io.File(simDir, "dfi_resp.v")
    // Always rewrite (probe/stub rule).
    val w = new java.io.PrintWriter(f)
    try w.write(dfiResp) finally w.close()
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
    // Always rewrite (probe/stub rule): the wrapper carries test-only
    // hacks (init_done force) that must track the source of truth here.
    val wwrap = new java.io.PrintWriter(wrap)
    try wwrap.write(ddrWrap) finally wwrap.close()
    simDir.getAbsolutePath
  }
}

/** Stage-A suite: JEDEC init only (green, run on demand as regression).
 * Run: `mill ... testOnly spinalML.memory.litedram.LiteDramCoreInitTest`
 * (or `cli/main.py test-all -k LiteDramCoreInit`). */
class LiteDramCoreInitTest extends AnyFunSuite {

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
      // 54 MHz board clock into the stub (sys2x=54/sys=27 exact).
      dut.clockDomain.forkStimulus(period = 18519)

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
}

// Stage-B suite (LiteDramCoreAxiTest) moved to LiteDramCoreAxiTest.scala
// (same package) to keep this file to GowinSimPrep + stage A.

// Stage-D suite (DfiResp + LiteDramCoreDfiTest) moved to
// LiteDramCoreDfiTest.scala (same package) to keep this file to
// GowinSimPrep + stage A.
