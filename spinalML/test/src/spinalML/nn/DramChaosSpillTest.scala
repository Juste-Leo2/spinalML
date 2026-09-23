// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axi.{Axi4, Axi4Config}
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.{BF16, I8}
import spinalML.harness.{DramChaosConfig, DramChaosInterposer, MemoryHarness, UniversalTestHarness}
import spinalML.replica.{HWArithmetic, ModelReplica, WeightMemoryLayout}

/**
 * S2 — DRAM chaos on spilling GEMM layers (Phase A, docs/ddr_stress.md).
 *
 * The DUT talks to `AxiMemorySim` through the test-only `DramChaosInterposer`
 * (LFSR read/write latency, row-miss + turnaround penalties, periodic refresh
 * blackouts, strict per-ID order). The oracle is the untouched `ModelReplica`:
 * bit-exactness must hold identically with and without chaos — any deviation
 * is a DUT timing-assumption bug, never a replica bug.
 *
 * P1-6: the top and the case are parameterized by (spec, inShape) so the same
 * chaos proof covers the spilling Linear (S2/P0c) and the spilling Conv2D
 * (P1-6, [6,6,2] in, K=2 -> [5,5,2] out, KFull=8, Ks=4, P=2). Chaos is the
 * harshest exerciser of the S2e single-beat seed pacing: blackouts and
 * row-miss penalties stall chunk acceptance mid-pass.
 */
class DramChaosSpillTop(val isInt: Boolean, val cfg: DramChaosConfig,
  val spec: Seq[LayerSpec], val inShape: Seq[Int]) extends Component {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val dt: Data = if (isInt) I8() else BF16()
  val dut = new Accelerator(
    dataType = dt,
    inputShape = inShape,
    modelSpec = spec,
    axiConfig = axiConfig,
    temporal = 1,
    memory = MemorySpec(spillBase = Some(0x30000L), capacityBytes = Some(4096))
  )
  val io = new Bundle {
    // Same-tag rule (Spinal flips the child side in `<>`): each top port
    // carries the same master/slave tag as the sub-block it continues.
    // Undriven boundary signals are poked by the sim threads (same as the
    // flat-DUT tests poking outStream.ready).
    val memPort = master(Axi4(axiConfig))
    val ctrlBus = slave(AxiLite4(dut.axiLiteConfig))
    val outStream = master(cloneOf(dut.io.outStream))
  }
  val chaos = DramChaosInterposer(axiConfig, cfg)
  chaos.io.dutSide <> dut.io.axiMaster
  chaos.io.memSide <> io.memPort
  io.ctrlBus <> dut.io.ctrlBus
  io.outStream <> dut.io.outStream
}

class DramChaosSpillTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val imgBase = 0x10000L
  val weightBase = 0x40000L
  val spillBase = 0x30000L

  private def writeWords(mem: spinal.lib.bus.amba4.axi.sim.SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)
  }

  def runChaosCase(isInt: Boolean, cfg: DramChaosConfig, label: String,
    spec: Seq[LayerSpec] = Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
    inShape: Seq[Int] = Seq(1, 8)): Unit = {
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    var packedWords: Seq[BigInt] = null
    var imgWords: Seq[BigInt] = null
    var expected: Seq[Double] = null

    // Untouched oracle + DDR image (same deterministic build as R3).
    SpinalConfig(targetDirectory = "out/tmp-chaos-spill-e2e").generateVerilog(new Component {
      setDefinitionName("ChaosSpillE2eLayout")
      val dt = if (isInt) I8() else BF16()
      val packed = WeightMemoryLayout.buildDeterministicWeights(spec, dt, axiConfig)
      packedWords = packed.words
      val k = inShape.product
      if (isInt) {
        val inInts = (0 until k).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
        imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        expected = ModelReplica.forwardWithTrace(
          spec, inShape, ModelReplica.IntTensor(inShape, inInts, 8), packed).logits
      } else {
        val inVals = (0 until k).map { idx =>
          val sign = if (idx % 2 == 0) 1.0 else -1.0
          HWArithmetic.fromDouble(sign * (((idx % 7) + 1) * 0.125), 8, 7)
        }
        val inFloats = inVals.map(f => HWArithmetic.decode(f, 8, 7).toFloat)
        imgWords = MemoryHarness.packFloats(MemoryHarness.padded(inFloats))
        expected = ModelReplica.forwardWithTrace(
          spec, inShape, ModelReplica.FloatTensor(inShape, inVals, 8, 7), packed).logits
      }
    })

    val compiled = SimConfig.withWave.compile(new DramChaosSpillTop(isInt, cfg, spec, inShape))
    compiled.doSim { top =>
      val dut = top.dut
      dut.clockDomain.forkStimulus(10)
      def tick(): Unit = {
        val _ = top.io.outStream.stream.valid.toBoolean
        dut.clockDomain.waitSampling()
      }

      val memSim = AxiMemorySim(
        axi = top.io.memPort,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memSim.start()

      writeWords(memSim.memory, weightBase, packedWords)
      writeWords(memSim.memory, imgBase, imgWords)

      def writeCsr(addr: BigInt, data: BigInt): Unit = {
        top.io.ctrlBus.aw.valid #= true
        top.io.ctrlBus.aw.payload.addr #= addr
        top.io.ctrlBus.w.valid #= true
        top.io.ctrlBus.w.payload.data #= data
        top.io.ctrlBus.w.payload.strb #= 0xF
        top.io.ctrlBus.b.ready #= true
        dut.clockDomain.waitSamplingWhere(top.io.ctrlBus.aw.ready.toBoolean && top.io.ctrlBus.w.ready.toBoolean)
        top.io.ctrlBus.aw.valid #= false
        top.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSamplingWhere(top.io.ctrlBus.b.valid.toBoolean)
        top.io.ctrlBus.b.ready #= false
        dut.clockDomain.waitSampling()
      }

      top.io.ctrlBus.aw.valid #= false
      top.io.ctrlBus.w.valid #= false
      top.io.ctrlBus.ar.valid #= false
      top.io.ctrlBus.b.ready #= false
      top.io.ctrlBus.r.ready #= false
      top.io.outStream.stream.ready #= true
      tick(); tick()

      var settled = 0
      var sc = 0
      while (settled < 5 && sc < 500) {
        val bLow = !top.io.memPort.b.valid.toBoolean
        val awR = top.io.memPort.aw.ready.toBoolean
        val arR = top.io.memPort.ar.ready.toBoolean
        if (bLow && awR && arR) settled += 1 else settled = 0
        tick(); sc += 1
      }
      assert(settled == 5, "memory-model agents never settled on the bus")

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      writeCsr(0x00, 1)

      // Probes on the mem side of the chaos (beats are preserved through the
      // gates, only timing shifts — counts equal the DUT-side traffic).
      // NOTE: dut.io.axiMaster is internal to this top and its dead cone is
      // trimmed by Verilator (memPort outputs have no RTL driver, sim pokes
      // only) — probe the kept top-level memPort instead.
      var arBeats = 0L
      var awBeats = 0L
      val outCount = top.io.outStream.shape.product
      val collected = scala.collection.mutable.ArrayBuffer[Double]()
      var cycles = 0
      val timeout = 200000
      while (collected.length < outCount && cycles < timeout) {
        if (top.io.memPort.ar.valid.toBoolean && top.io.memPort.ar.ready.toBoolean)
          arBeats += top.io.memPort.ar.payload.len.toInt + 1
        if (top.io.memPort.aw.valid.toBoolean && top.io.memPort.aw.ready.toBoolean)
          awBeats += top.io.memPort.aw.payload.len.toInt + 1
        if (top.io.outStream.stream.valid.toBoolean) {
          for (l <- 0 until top.io.outStream.lanes if collected.length < outCount) {
            if (isInt) collected += top.io.outStream.stream.payload(l).asInstanceOf[SInt].toInt.toDouble
            else collected += UniversalTestHarness.decodeFloat(top.io.outStream.stream.payload(l)).toDouble
          }
        }
        tick(); cycles += 1
      }
      assert(collected.length == outCount,
        s"[$label] CHAOS TIMEOUT/DEADLOCK: collected ${collected.length}/$outCount in $cycles cycles")

      val devs = collected.toSeq.zip(expected).map { case (h, s) => scala.math.abs(h - s) }
      val dev = devs.max
      devs.zipWithIndex.foreach { case (d, i) =>
        if (d != 0.0) println(f"S2 [$label DEV] out[$i] hw=${collected(i)}%9.5f sw=${expected(i)}%9.5f dev=$d%9.6f")
      }
      assert(dev == 0.0, s"S2 [$label] chaos bit-exact failed: max dev=$dev")
      println(s"S2 [$label] chaos bit-exact PASSED in $cycles cycles (AR beats=$arBeats, AW beats=$awBeats)")
    }
  }

  test("S2 chaos-heavy I8 spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.heavy(), label = "CHAOS-I8-K8-P2")
  }

  test("S2 chaos-heavy BF16 spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.heavy(), label = "CHAOS-BF16-K8-P2")
  }

  test("P0c chaos-light I8 spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.light(), label = "CHAOS-LIGHT-I8-K8-P2")
  }

  test("P0c chaos-light BF16 spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.light(), label = "CHAOS-LIGHT-BF16-K8-P2")
  }

  test("P0c chaos-heavy seed sweep I8 spill P=2 bit-exact vs ModelReplica") {
    // Two extra deterministic seeds: no seed-dependent deadlock/deviation.
    runChaosCase(isInt = true, DramChaosConfig.heavy(0xBEEF01L), label = "CHAOS-SEED1-I8-K8-P2")
    runChaosCase(isInt = true, DramChaosConfig.heavy(0x123456L), label = "CHAOS-SEED2-I8-K8-P2")
  }

  // P1-6 chaos on the spilling Conv2D ([6,6,2] in, K=2 -> [5,5,2] out,
  // KFull=8, Ks=4, P=2 — same geometry as ConvReplicaSpillTest).
  val convSpillSpec = Seq(Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 4))
  val convSpillShape = Seq(6, 6, 2)

  test("P1-6 chaos-heavy I8 conv spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.heavy(), label = "CHAOS-CONV-I8-P2",
      spec = convSpillSpec, inShape = convSpillShape)
  }

  test("P1-6 chaos-heavy BF16 conv spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.heavy(), label = "CHAOS-CONV-BF16-P2",
      spec = convSpillSpec, inShape = convSpillShape)
  }

  test("P1-6 chaos-light I8 conv spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.light(), label = "CHAOS-CONV-LIGHT-I8-P2",
      spec = convSpillSpec, inShape = convSpillShape)
  }

  test("P1-6 chaos-light BF16 conv spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.light(), label = "CHAOS-CONV-LIGHT-BF16-P2",
      spec = convSpillSpec, inShape = convSpillShape)
  }

  // P2-6 chaos on the spilling Conv1D ([6,4] in, K=2 -> [5,2] out,
  // KFull=8, Ks=4, P=2 — same geometry as Conv1DReplicaSpillTest).
  val conv1DSpillSpec = Seq(Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2,
    weightLanes = 4, spillKSlice = 4))
  val conv1DSpillShape = Seq(6, 4)

  test("P2-6 chaos-heavy I8 conv1d spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.heavy(), label = "CHAOS-CONV1D-I8-P2",
      spec = conv1DSpillSpec, inShape = conv1DSpillShape)
  }

  test("P2-6 chaos-heavy BF16 conv1d spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.heavy(), label = "CHAOS-CONV1D-BF16-P2",
      spec = conv1DSpillSpec, inShape = conv1DSpillShape)
  }

  test("P2-6 chaos-light I8 conv1d spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = true, DramChaosConfig.light(), label = "CHAOS-CONV1D-LIGHT-I8-P2",
      spec = conv1DSpillSpec, inShape = conv1DSpillShape)
  }

  test("P2-6 chaos-light BF16 conv1d spill P=2 bit-exact vs ModelReplica") {
    runChaosCase(isInt = false, DramChaosConfig.light(), label = "CHAOS-CONV1D-LIGHT-BF16-P2",
      spec = conv1DSpillSpec, inShape = conv1DSpillShape)
  }
}
