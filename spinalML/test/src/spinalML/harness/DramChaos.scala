// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.harness

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi.{Axi4, Axi4Config}
import spinal.lib.bus.amba4.axilite.AxiLite4

/**
 * Interface a stress-test top must expose for `UniversalTestHarness.runStress`.
 *
 * The generated CLI scaffold wraps any user DUT in such a top (DUT +
 * `DramChaosInterposer` + exposed ports); the harness stays generic by
 * programming against this trait instead of a concrete wrapper type.
 * Concrete port types only — no timing/config leaks into the oracle path.
 */
trait ChaosDut {
  def memAxi: Axi4
  def ctrlAxi: AxiLite4
  def outShape: Seq[Int]
  def outLanes: Int
  def setOutReady(v: Boolean): Unit
  def outValid: Boolean
  def outPayload(lane: Int): Data
}

/**
 * DRAM chaos model (Phase A, docs/ddr_stress.md) — TEST-ONLY interposer.
 *
 * Sits between `dut.io.axiMaster` and the functional memory model
 * (`AxiMemorySim`) and injects DRAM-class timing pressure while preserving
 * strict AXI legality:
 * - per-channel latency (LFSR jitter, deterministic per transaction count);
 * - row hit/miss penalty from AR address history (`rowBits` row size);
 * - read/write turnaround penalty;
 * - periodic refresh blackouts (outputs stall, FIFOs absorb, backpressure
 *   propagates — exactly like a real controller);
 * - strict per-channel FIFO order: same-ID order is NEVER violated
 *   (a real controller guarantees per-ID order too), only inter-ID
 *   interleave varies via independent per-channel delays.
 *
 * The software oracle (`ModelReplica`) never sees this block: bit-exactness
 * must hold identically with and without chaos, otherwise the DUT has a
 * timing-assumption bug (not a replica bug).
 */
case class DramChaosConfig(
  enable: Boolean = false,
  seed: Long = 0xC4A05L,
  // Base per-beat delays [min, max] (LFSR-uniform, sampled per transaction).
  arDelayMin: Int = 0, arDelayMax: Int = 0,
  rDelayMin: Int = 0, rDelayMax: Int = 0,
  awDelayMin: Int = 0, awDelayMax: Int = 0,
  wDelayMin: Int = 0, wDelayMax: Int = 0,
  bDelayMin: Int = 0, bDelayMax: Int = 0,
  // Refresh: every `refreshPeriod` cycles, outputs stall `refreshBlackout`
  // cycles. refreshPeriod <= 0 disables refresh.
  refreshPeriod: Int = 0,
  refreshBlackout: Int = 0,
  // Row model: row = addr >> rowBits. A read to a new row costs
  // `rowMissPenalty` extra cycles on AR; switching direction costs
  // `turnaroundPenalty` (applied on AR after a write).
  rowBits: Int = 10,
  rowMissPenalty: Int = 0,
  turnaroundPenalty: Int = 0,
  fifoDepth: Int = 16
) {
  require(refreshBlackout <= refreshPeriod,
    s"DramChaosConfig: refreshBlackout=$refreshBlackout > refreshPeriod=$refreshPeriod")
  require(arDelayMax >= arDelayMin && rDelayMax >= rDelayMin &&
    awDelayMax >= awDelayMin && wDelayMax >= wDelayMin && bDelayMax >= bDelayMin,
    "DramChaosConfig: delay max < min")
}

object DramChaosConfig {
  val disabled: DramChaosConfig = DramChaosConfig()

  def light(seed: Long = 0xCAFE01L): DramChaosConfig = DramChaosConfig(
    enable = true, seed = seed,
    arDelayMin = 1, arDelayMax = 3,
    rDelayMin = 0, rDelayMax = 1,
    awDelayMin = 1, awDelayMax = 2,
    wDelayMin = 0, wDelayMax = 1,
    bDelayMin = 1, bDelayMax = 2,
    rowMissPenalty = 2, turnaroundPenalty = 2
  )

  def heavy(seed: Long = 0xCAFE02L): DramChaosConfig = DramChaosConfig(
    enable = true, seed = seed,
    arDelayMin = 2, arDelayMax = 8,
    rDelayMin = 0, rDelayMax = 3,
    awDelayMin = 2, awDelayMax = 6,
    wDelayMin = 0, wDelayMax = 2,
    bDelayMin = 2, bDelayMax = 6,
    refreshPeriod = 200, refreshBlackout = 12,
    rowMissPenalty = 6, turnaroundPenalty = 4
  )
}

/** One delayed Stream gate: data FIFO + release-time FIFO, LFSR hold. */
case class ChaosGate(width: Int, dMin: Int, dMax: Int, seed: Long, timeWidth: Int = 24) extends Component {
  val io = new Bundle {
    val sink = slave(Stream(Bits(width bits)))
    val source = master(Stream(Bits(width bits)))
    val now = in UInt(timeWidth bits)
    val penalty = in UInt(8 bits)
    val blackout = in Bool()
  }

  val dataF = StreamFifo(Bits(width bits), 16)
  val relF = StreamFifo(UInt(timeWidth bits), 16)
  dataF.io.push << io.sink

  // 16-bit Galois LFSR, advanced per pushed transaction (deterministic in the
  // transaction count, independent of wall cycles).
  val lfsr = Reg(Bits(16 bits)) init(((seed & 0xFFFF) | 1).toInt)
  val span = dMax - dMin + 1
  require(span >= 1, s"ChaosGate: dMax=$dMax < dMin=$dMin")
  val extra: UInt = if (span > 1) (lfsr.asUInt % U(span, 16 bits)).resize(8 bits) else U(0, 8 bits)

  relF.io.push.valid := dataF.io.push.fire
  relF.io.push.payload := (io.now + dMin + extra + io.penalty).resize(timeWidth bits)
  when(dataF.io.push.fire) {
    lfsr := ((lfsr(0) ^ lfsr(2) ^ lfsr(3) ^ lfsr(5)) ## lfsr(15 downto 1))
  }

  val due = relF.io.pop.valid && relF.io.pop.payload <= io.now
  io.source.valid := dataF.io.pop.valid && due && !io.blackout
  io.source.payload := dataF.io.pop.payload
  val fire = io.source.valid && io.source.ready
  dataF.io.pop.ready := fire
  relF.io.pop.ready := fire
}

case class DramChaosInterposer(axiConfig: Axi4Config, cfg: DramChaosConfig) extends Component {
  val io = new Bundle {
    val dutSide = slave(Axi4(axiConfig))
    val memSide = master(Axi4(axiConfig))
  }

  if (!cfg.enable) {
    // Bypass: combinationally transparent, for harness plumbing only.
    io.memSide.ar <> io.dutSide.ar
    io.memSide.aw <> io.dutSide.aw
    io.memSide.w <> io.dutSide.w
    io.dutSide.r <> io.memSide.r
    io.dutSide.b <> io.memSide.b
  } else {
    val timeWidth = 24
    val now = Reg(UInt(timeWidth bits)) init(0)
    now := now + 1

    // Refresh blackout window.
    val blackout: Bool = if (cfg.refreshPeriod > 0) {
      val cnt = Reg(UInt(log2Up(cfg.refreshPeriod) bits)) init(0)
      when(cnt === cfg.refreshPeriod - 1) { cnt := 0 } otherwise { cnt := cnt + 1 }
      cnt >= U(cfg.refreshPeriod - cfg.refreshBlackout, log2Up(cfg.refreshPeriod) bits)
    } else False

    // Address-history row tracker (AR plane): row = addr >> rowBits.
    val rowW = axiConfig.addressWidth - cfg.rowBits
    val lastRow = Reg(UInt(rowW bits)) init(0)
    val lastWasWrite = Reg(Bool()) init(False)
    val firstDone = Reg(Bool()) init(False)
    val arRow = io.dutSide.ar.payload.addr(axiConfig.addressWidth - 1 downto cfg.rowBits)
    val rowHit = firstDone && arRow === lastRow
    val arPenalty =
      Mux(!firstDone, U(0, 8 bits),
        Mux(rowHit, U(0, 8 bits), U(cfg.rowMissPenalty, 8 bits)) +
          Mux(!rowHit && lastWasWrite, U(cfg.turnaroundPenalty, 8 bits), U(0, 8 bits)))
    when(io.dutSide.ar.fire) {
      lastRow := arRow
      lastWasWrite := False
      firstDone := True
    }
    when(io.dutSide.aw.fire) {
      lastWasWrite := True
    }

    // AR: dut -> mem, row/turnaround penalty.
    val gAr = ChaosGate(io.dutSide.ar.payload.getBitsWidth, cfg.arDelayMin, cfg.arDelayMax, cfg.seed + 1)
    gAr.io.now := now
    gAr.io.penalty := arPenalty
    gAr.io.blackout := blackout
    gAr.io.sink.valid := io.dutSide.ar.valid
    gAr.io.sink.payload := io.dutSide.ar.payload.asBits
    io.dutSide.ar.ready := gAr.io.sink.ready
    io.memSide.ar.valid := gAr.io.source.valid
    io.memSide.ar.payload.assignFromBits(gAr.io.source.payload)
    gAr.io.source.ready := io.memSide.ar.ready

    // R: mem -> dut, beat jitter only (order preserved by FIFO).
    val gR = ChaosGate(io.memSide.r.payload.getBitsWidth, cfg.rDelayMin, cfg.rDelayMax, cfg.seed + 2)
    gR.io.now := now
    gR.io.penalty := U(0, 8 bits)
    gR.io.blackout := blackout
    gR.io.sink.valid := io.memSide.r.valid
    gR.io.sink.payload := io.memSide.r.payload.asBits
    io.memSide.r.ready := gR.io.sink.ready
    io.dutSide.r.valid := gR.io.source.valid
    io.dutSide.r.payload.assignFromBits(gR.io.source.payload)
    gR.io.source.ready := io.dutSide.r.ready

    // AW: dut -> mem.
    val gAw = ChaosGate(io.dutSide.aw.payload.getBitsWidth, cfg.awDelayMin, cfg.awDelayMax, cfg.seed + 3)
    gAw.io.now := now
    gAw.io.penalty := U(0, 8 bits)
    gAw.io.blackout := blackout
    gAw.io.sink.valid := io.dutSide.aw.valid
    gAw.io.sink.payload := io.dutSide.aw.payload.asBits
    io.dutSide.aw.ready := gAw.io.sink.ready
    io.memSide.aw.valid := gAw.io.source.valid
    io.memSide.aw.payload.assignFromBits(gAw.io.source.payload)
    gAw.io.source.ready := io.memSide.aw.ready

    // W: dut -> mem.
    val gW = ChaosGate(io.dutSide.w.payload.getBitsWidth, cfg.wDelayMin, cfg.wDelayMax, cfg.seed + 4)
    gW.io.now := now
    gW.io.penalty := U(0, 8 bits)
    gW.io.blackout := blackout
    gW.io.sink.valid := io.dutSide.w.valid
    gW.io.sink.payload := io.dutSide.w.payload.asBits
    io.dutSide.w.ready := gW.io.sink.ready
    io.memSide.w.valid := gW.io.source.valid
    io.memSide.w.payload.assignFromBits(gW.io.source.payload)
    gW.io.source.ready := io.memSide.w.ready

    // B: mem -> dut.
    val gB = ChaosGate(io.memSide.b.payload.getBitsWidth, cfg.bDelayMin, cfg.bDelayMax, cfg.seed + 5)
    gB.io.now := now
    gB.io.penalty := U(0, 8 bits)
    gB.io.blackout := blackout
    gB.io.sink.valid := io.memSide.b.valid
    gB.io.sink.payload := io.memSide.b.payload.asBits
    io.memSide.b.ready := gB.io.sink.ready
    io.dutSide.b.valid := gB.io.source.valid
    io.dutSide.b.payload.assignFromBits(gB.io.source.payload)
    gB.io.source.ready := io.dutSide.b.ready
  }
}
