// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.{BF16, I8}
import spinalML.harness.{MemoryHarness, UniversalTestHarness}
import spinalML.replica.{HWArithmetic, ModelReplica, WeightMemoryLayout}

/**
 * R3 end-to-end: a spilling Linear inside a real Sequential + Accelerator,
 * DDR-backed throughout, bit-exact against the full `ModelReplica` oracle
 * (layout R1 + fold R2) — closing the S3 "replica spill" gap
 * (docs/ddr_final_impl.md). The hand-oracle region proofs stay in
 * `SequentialSpillTest`; here the oracle is the replica itself.
 *
 * Cases: I8 P=2 (exact int math) and BF16 P=2 (bit-exact fadd order).
 * Both also report AXI traffic (R5: measured `P×(W_slice+A+2·M·N)` input).
 */
class SequentialReplicaSpillTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  val imgBase = 0x10000L
  val weightBase = 0x40000L
  val spillBase = 0x30000L

  private def writeWords(mem: spinal.lib.bus.amba4.axi.sim.SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)
  }

  /** Generic replica-oracle e2e for a single spilling Linear. */
  def runReplicaCase(
    spec: Seq[LayerSpec],
    inShape: Seq[Int],
    isInt: Boolean,
    label: String
  ): Unit = {
    var packedWords: Seq[BigInt] = null
    var imgWords: Seq[BigInt] = null
    var expected: Seq[Double] = null
    var weightBytesTotal = 0

    val compiled = SimConfig.withWave.compile({
      val dt: Data = if (isInt) I8() else BF16()
      new Accelerator(
        dataType = dt,
        inputShape = inShape,
        modelSpec = spec,
        axiConfig = axiConfig,
        temporal = 1,
        memory = MemorySpec(spillBase = Some(spillBase), capacityBytes = Some(4096))
      )
    })

    // Oracle + DDR image built inside elaboration (dtypes need context).
    // NOTE: recompiles per case via the compiled DUT below — layout runs twice
    // (here for words, and dut.globalDataType read is not available pre-sim),
    // so rebuild deterministically with the same dtype idiom.
    SpinalConfig(targetDirectory = "out/tmp-replica-spill-e2e").generateVerilog(new Component {
      setDefinitionName("ReplicaSpillE2eLayout")
      val dt = if (isInt) I8() else BF16()
      val packed = WeightMemoryLayout.buildDeterministicWeights(spec, dt, axiConfig)
      packedWords = packed.words
      weightBytesTotal = packed.totalBytes
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
        // BF16 16-bit packing via the CLI scaffold idiom (exact: inputs are
        // BF16-representable, so float32 round-trips bit-identically).
        val inFloats = inVals.map(f => HWArithmetic.decode(f, 8, 7).toFloat)
        imgWords = MemoryHarness.packFloats(MemoryHarness.padded(inFloats))
        expected = ModelReplica.forwardWithTrace(
          spec, inShape, ModelReplica.FloatTensor(inShape, inVals, 8, 7), packed).logits
      }
    })

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      def tick(): Unit = {
        val _ = dut.io.outStream.stream.valid.toBoolean
        dut.clockDomain.waitSampling()
      }

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memSim.start()

      writeWords(memSim.memory, weightBase, packedWords)
      writeWords(memSim.memory, imgBase, imgWords)

      def writeCsr(addr: BigInt, data: BigInt): Unit = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addr
        dut.io.ctrlBus.w.valid #= true
        dut.io.ctrlBus.w.payload.data #= data
        dut.io.ctrlBus.w.payload.strb #= 0xF
        dut.io.ctrlBus.b.ready #= true
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
        dut.io.ctrlBus.b.ready #= false
        dut.clockDomain.waitSampling()
      }
      def readCsr(addr: BigInt): BigInt = {
        dut.io.ctrlBus.ar.valid #= true
        dut.io.ctrlBus.ar.payload.addr #= addr
        dut.io.ctrlBus.r.ready #= true
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.ar.ready.toBoolean)
        dut.io.ctrlBus.ar.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.r.valid.toBoolean)
        val data = dut.io.ctrlBus.r.payload.data.toBigInt
        dut.io.ctrlBus.r.ready #= false
        dut.clockDomain.waitSampling()
        data
      }

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      tick(); tick()

      // Bring-up settle (S2b lesson): memory-model agents own the bus first.
      var settled = 0
      var sc = 0
      while (settled < 5 && sc < 500) {
        val bLow = !dut.io.axiMaster.b.valid.toBoolean
        val awR = dut.io.axiMaster.aw.ready.toBoolean
        val arR = dut.io.axiMaster.ar.ready.toBoolean
        if (bLow && awR && arR) settled += 1 else settled = 0
        tick(); sc += 1
      }
      assert(settled == 5, "memory-model agents never settled on the bus")

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      // 0x34 spill base: descriptor init path under test, no CSR write.
      writeCsr(0x00, 1)

      // R5 traffic probes: count AXI command beats (fire cycles x len+1).
      var arBeats = 0L
      var awBeats = 0L
      val outCount = dut.io.outStream.shape.product
      val collected = scala.collection.mutable.ArrayBuffer[Double]()
      var cycles = 0
      val timeout = 50000
      while (collected.length < outCount && cycles < timeout) {
        if (dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean)
          arBeats += dut.io.axiMaster.ar.payload.len.toInt + 1
        if (dut.io.axiMaster.aw.valid.toBoolean && dut.io.axiMaster.aw.ready.toBoolean)
          awBeats += dut.io.axiMaster.aw.payload.len.toInt + 1
        if (dut.io.outStream.stream.valid.toBoolean) {
          for (l <- 0 until dut.io.outStream.lanes if collected.length < outCount) {
            if (isInt) collected += dut.io.outStream.stream.payload(l).asInstanceOf[SInt].toInt.toDouble
            else collected += UniversalTestHarness.decodeFloat(dut.io.outStream.stream.payload(l)).toDouble
          }
        }
        tick(); cycles += 1
      }
      assert(collected.length == outCount,
        s"[$label] collected ${collected.length}/$outCount outputs in $cycles cycles")

      val devs = collected.toSeq.zip(expected).map { case (h, s) => math.abs(h - s) }
      val dev = devs.max
      devs.zipWithIndex.foreach { case (d, i) =>
        if (d != 0.0) println(f"R3 [$label DEV] out[$i] hw=${collected(i)}%9.5f sw=${expected(i)}%9.5f dev=$d%9.6f")
      }
      assert(dev == 0.0, s"R3 [$label] replica bit-exact failed: max dev=$dev")
      println(s"R3 [$label] replica bit-exact PASSED in $cycles cycles " +
        s"(AR beats=$arBeats, AW beats=$awBeats, weightRegionBytes=$weightBytesTotal)")

      val tileCnt = readCsr(CsrMap.TileCnt)
      assert(tileCnt == 1, s"TILE_CNT=$tileCnt (expected 1 frame)")
      val status = readCsr(CsrMap.Status)
      assert(status == 0, s"status=0x${status.toString(16)} (expected idle STOP)")
      val mode = readCsr(CsrMap.Mode)
      assert(mode == 0, s"MODE=0x${mode.toString(16)} (expected STREAM_PER_PASS)")
      val spillRb = readCsr(CsrMap.SpillBase)
      assert(spillRb == spillBase, f"CSR 0x34=0x$spillRb%X (expected 0x$spillBase%X)")
    }
  }

  test("R3 e2e I8 spill P=2 bit-exact vs ModelReplica") {
    runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), isInt = true, label = "I8-K8-P2")
  }

  test("R3 e2e BF16 spill P=2 bit-exact vs ModelReplica") {
    runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), isInt = false, label = "BF16-K8-P2")
  }
}
