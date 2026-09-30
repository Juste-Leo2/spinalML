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
      val awLen = in UInt(8 bits)
      val wValid = in Bool()
      val wData = in Bits(64 bits)
      val wStrb = in Bits(8 bits)
      val wFirst = in Bool()
      val wLast = in Bool()
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
    core.io.aw_payload_len := io.awLen
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
    core.io.w_first := io.wFirst
    core.io.w_last := io.wLast
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
      dut.clockDomain.forkStimulus(period = 18519) // 54 MHz into the stub (sys2x=54/sys=27 exact)
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

      // SoC-like INCR burst (mimics DMAWriter): one address phase for N
      // beats; last=1 ONLY on the final beat so the converter merges
      // consecutive half-words into full 128-bit words instead of
      // committing each half zero-padded (which clobbers).
      // CRITICAL: wValid stays HIGH across beats (no gaps!) with data
      // changing after each accept. Dropping wValid between beats inserts
      // idle cycles that defeat the converter merge (each beat commits
      // separately, zero-padded, clobbering its pair).
      def axiWriteBurst(wordAddr: BigInt, datas: Seq[BigInt]): Unit = {
        dut.io.awAddr #= wordAddr
        dut.io.awLen #= datas.size - 1
        dut.io.awValid #= true
        // 2 TB cycles = 37ns = 1 sys period: the window always contains
        // exactly one sys edge (exactly-once), any phase.
        cd.waitSampling(2)
        dut.io.awValid #= false
        dut.io.wValid #= true
        for ((d, i) <- datas.zipWithIndex) {
          dut.io.wData #= d
          dut.io.wStrb #= 0xFF
          dut.io.wFirst #= (i == 0)
          dut.io.wLast #= (i == datas.size - 1)
          // Same 2-cycle minimum per beat (window ~= 1 sys period),
          // valid held across beats (no gaps: converter merge needs it).
          var wtw = 0
          var wacc = false
          while ((wtw < 2 || !wacc) && wtw < 30000) {
            cd.waitSampling(1); wtw += 1
            if (dut.io.wReady.toBoolean) wacc = true
          }
          assert(wacc, f"w_ready timeout @word $wordAddr%x beat $i")
        }
        dut.io.wValid #= false
        var wt = 1
        var bSeen = false
        var bRespVal = BigInt(-1)
        while (wt < 30000 && !bSeen) {
          cd.waitSampling(1); wt += 1
          if (dut.io.bValid.toBoolean) { bSeen = true; bRespVal = dut.io.bResp.toBigInt }
        }
        assert(bSeen, f"write b_valid timeout @word $wordAddr%x")
        assert(bRespVal == 0, f"write RESP not OKAY (got $bRespVal)")
        cd.waitSampling(5)
        // Reopen check (see stage B): pipe must be free again.
        wt = 0
        while (!dut.io.awReady.toBoolean && wt < 2000) { cd.waitSampling(1); wt += 1 }
        assert(dut.io.awReady.toBoolean, f"aw never reopened @word $wordAddr%x (burst2beat stuck?)")
      }

      def axiRead(wordAddr: BigInt): BigInt = {
        dut.io.arAddr #= wordAddr
        dut.io.arValid #= true
        // Fire-and-forget (see axiWriteBurst): 2-cycle window.
        cd.waitSampling(2)
        dut.io.arValid #= false
        var n = 1
        var rSeen = false
        var rDataVal = BigInt(0)
        while (n < 30000 && !rSeen) {
          cd.waitSampling(1); n += 1
          if (dut.io.rValid.toBoolean) { rSeen = true; rDataVal = dut.io.rData.toBigInt }
        }
        assert(rSeen, f"read r_valid timeout @word $wordAddr%x (PHY read hang?)")
        cd.waitSampling(5)
        rDataVal
      }

      // One 2-beat burst: words 0x0+0x1 = one 128-bit DRAM word, merged
      // by the converter (no PRE/ACT/auto_precharge in the way).
      val d1 = BigInt("0123456789ABCDEF", 16)
      val d2 = BigInt("FEDCBA9876543210", 16)
      axiWriteBurst(BigInt(0x0), Seq(d1, d2))
      val got0 = axiRead(BigInt(0x0))
      assert(got0 == d1, f"mismatch @word 0: got $got0%016X exp $d1%016X")
      val got1 = axiRead(BigInt(0x1))
      assert(got1 == d2, f"mismatch @word 1: got $got1%016X exp $d2%016X")
    }
  }
}
