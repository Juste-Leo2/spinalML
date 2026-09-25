// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{Axi4ReadOnlySlaveAgent, AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.{BF16, FloatML, I4, I8}
import spinalML.harness.MemoryHarness
import spinalML.replica.{HWArithmetic, ModelReplica, WeightMemoryLayout}

class SequentialTest extends AnyFunSuite {
  /** Sequential SoC integration (non-spill + residency/prefetch): consecutive
   * inferences, band tiling, AR traffic metering, arbiter cascade and
   * idWidth fit — bit-exact vs ModelReplica throughout. */
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  val imgBase = 0x10000L
  val weightBase = 0x40000L

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) {
      mem.writeBigInt(base + i * 8, w, 8)
    }
  }

  private def decodeData(p: Data): Double = {
    p match {
      case f: FloatML =>
        val eW = f.exponent.getWidth
        val mW = f.mantissa.getWidth
        if (eW == 8 && mW == 7) {
          val bits = ((if (f.sign.toBoolean) 1 else 0) << 15) | ((f.exponent.toInt & 0xFF) << 7) | (f.mantissa.toInt & 0x7F)
          java.lang.Float.intBitsToFloat(bits << 16).toDouble
        } else {
          val sign = if (f.sign.toBoolean) -1.0 else 1.0
          val rawE = f.exponent.toInt
          val rawM = f.mantissa.toInt
          val bias = (1 << (eW - 1)) - 1
          val mag =
            if (rawE == 0) rawM.toDouble * math.pow(2.0, 1 - bias - mW)
            else (1.0 + rawM.toDouble / (1 << mW)) * math.pow(2.0, rawE - bias)
          sign * mag
        }
      case s: SInt => s.toBigInt.toLong.toDouble
      case u: UInt => u.toBigInt.toLong.toDouble
      case b: Bits => b.toBigInt.toLong.toDouble
      case bt: BaseType => bt.toBigInt.toDouble
      case _ => throw new IllegalArgumentException(s"Unsupported data type: $p")
    }
  }

  test("Sequential: Consecutive inferences with state re-arming without reset") {
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3),
      ReLU(),
      Flatten(),
      Linear(inFeatures = 8, outFeatures = 2)
    )

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Accelerator(
        dataType = I8(),
        inputShape = Seq(4, 4, 1),
        modelSpec = spec,
        axiConfig = axiConfig
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memSim.start()

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

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

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

      val numPasses = 5
      for (pass <- 0 until numPasses) {
        val inInts = (0 until 16).map(idx => ((idx * 3 + pass * 5) % 7).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

        writeCsr(0x00, 1)

        val collected = scala.collection.mutable.ArrayBuffer[Double]()
        var cycles = 0
        val timeout = 10000

        while (collected.length < 2 && cycles < timeout) {
          if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
            collected += decodeData(dut.io.outStream.stream.payload(0))
          }
          dut.clockDomain.waitSampling()
          cycles += 1
        }

        assert(cycles < timeout, s"Pass $pass timed out")
        assert(collected.length == 2, s"Pass $pass: expected 2 outputs, got ${collected.length}")

        val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Pass $pass mismatch: HW $collected vs SW ${oracle.logits}")
        println(f"[SequentialTest] Pass $pass bit-exact (max dev = $dev%.3f in $cycles cycles)")
      }

      println(s"[SequentialTest] All $numPasses consecutive inferences passed cleanly without reset")
    }
  }

  test("Sequential: busy survives a START accepted on the final-beat cycle (NN-02)") {
    // NN-02: ioBusy is written by two `when` blocks; the clear (last output beat)
    // used to win over a same-cycle START accept, dropping busy while a new
    // inference had just begun. A single-beat frame (outFeatures=1) makes the
    // only output beat the final one: hold it with ready=0, then present the
    // new START and the completion on the very same DUT sampling edge.
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3),
      ReLU(),
      Flatten(),
      Linear(inFeatures = 8, outFeatures = 1)
    )

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Sequential(
        globalDataType = I8(),
        inputShape = Seq(4, 4, 1),
        layers = spec,
        axiConfig = axiConfig
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      // Sequential exposes a read-only AXI master: drive it with the read-only
      // slave agent over a plain SparseMemory.
      val memory = SparseMemory()
      new Axi4ReadOnlySlaveAgent(dut.io.axiMaster, dut.clockDomain) {
        override def readByte(address: BigInt, id: Int): Byte = memory.read(address.toLong)
      }

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.layers, dut.globalDataType, axiConfig)
      writeWords(memory, weightBase, packed.words)

      val inInts = (0 until 16).map(idx => ((idx * 3) % 7).toLong)
      writeWords(memory, imgBase, MemoryHarness.packBytes(inInts.map(_.toInt)))

      dut.io.start.valid #= false
      dut.io.outStream.stream.ready #= false
      dut.io.imgBaseAddress #= imgBase
      dut.io.weightsBaseAddress #= weightBase
      dut.clockDomain.waitSampling(5)

      val frameSize = (dut.finalShape.product + dut.finalLanes - 1) / dut.finalLanes
      assert(frameSize == 1, s"NN-02 harness expects a single-beat frame, got $frameSize")

      // Start the first inference. Its only (final) output beat is held by the
      // ready=0 set above, so the frame is exactly one fire from completion.
      dut.io.start.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.start.ready.toBoolean)
      dut.io.start.valid #= false
      dut.clockDomain.waitSamplingWhere(dut.io.outStream.stream.valid.toBoolean)

      assert(dut.io.busy.toBoolean, "ioBusy lost before the final beat (harness assumption broken)")
      assert(dut.io.start.ready.toBoolean,
        "START not accepted while holding the final beat (harness assumption broken)")

      // Same DUT edge: the new START is accepted AND the final beat completes.
      // Both inputs are assigned in the same delta, so they become visible on
      // the same sampling edge (two edges after the assignment in this sim).
      dut.io.start.valid #= true
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling()
      dut.clockDomain.waitSampling()

      val busyAfter = dut.io.busy.toBoolean
      val beatFired = !dut.io.outStream.stream.valid.toBoolean
      dut.io.start.valid #= false
      dut.io.outStream.stream.ready #= false

      assert(beatFired,
        "final beat did not complete on the conflict edge (harness assumption broken)")
      assert(busyAfter,
        "ioBusy fell while a new inference START was accepted on the final-beat cycle (NN-02)")
      println("[SequentialTest] NN-02: busy stayed high across the same-cycle START + final beat")
    }
  }

  test("Sequential: Vertical band tiling continuity on 64x64 W4A8 CNN (tileHeight=16, 4 bands)") {
    // 64x64 image with I8 activations and I4 narrow integer weights (W4A8)
    // Banded with tileHeight = 16 (4 bands total)
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3, customWeightType = Some(HardType(I4()))),
      ReLU(),
      MaxPool2D(poolSize = 2, stride = 2), // 62x62 -> 31x31 = 961 features
      Flatten(),
      Linear(inFeatures = 961, outFeatures = 2)
    )

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Accelerator(
        dataType = I8(),
        inputShape = Seq(64, 64, 1),
        modelSpec = spec,
        axiConfig = axiConfig,
        tileHeight = 16
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memSim.start()

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

      // 64*64 = 4096 elements in I8 = 4096 bytes = 512 AXI beats (64-bit)
      val totalPixels = 64 * 64
      val inInts = (0 until totalPixels).map(idx => ((idx % 7) - 3).toLong)

      val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      writeWords(memSim.memory, imgBase, imgWords)

      val inputTensor = ModelReplica.IntTensor(Seq(64, 64, 1), inInts, 8)
      val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

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

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      writeCsr(0x00, 1)

      val collected = scala.collection.mutable.ArrayBuffer[Double]()
      var cycles = 0
      val timeout = 100000

      while (collected.length < 2 && cycles < timeout) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          collected += decodeData(dut.io.outStream.stream.payload(0))
        }
        dut.clockDomain.waitSampling()
        cycles += 1
      }

      assert(cycles < timeout, s"64x64 I4 Band Tiling timed out after $cycles cycles")
      assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

      val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
      assert(dev == 0.0, s"Tiled HW $collected vs SW ${oracle.logits} (max dev = $dev)")
      println(f"[SequentialTest] 64x64 I4 Band Tiling (tileHeight=16, 4 bands) bit-exact: max dev = $dev%.3f in $cycles cycles")
    }
  }

  test("Sequential: Vertical band tiling continuity on 8x8 BF16 CNN (tileHeight=4, 2 bands)") {
    // 8x8 image with BF16 floating-point activations and weights
    // Banded with tileHeight = 4 (2 bands total: rows 0..3 and rows 4..7)
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 1, kernelSize = 3),
      ReLU(),
      MaxPool2D(poolSize = 2, stride = 2), // 6x6 -> 3x3 = 9 features
      Flatten(),
      Linear(inFeatures = 9, outFeatures = 2)
    )

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Accelerator(
        dataType = BF16(),
        inputShape = Seq(8, 8, 1),
        modelSpec = spec,
        axiConfig = axiConfig,
        tileHeight = 4
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memSim.start()

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

      // 8*8 = 64 elements in BF16
      val inFloats = (0 until 64).map(idx => (((idx % 7) - 3) * 0.25f))
      val imgWords = MemoryHarness.packFloats(MemoryHarness.padded(inFloats))
      writeWords(memSim.memory, imgBase, imgWords)

      val inputTensor = ModelReplica.FloatTensor(Seq(8, 8, 1), inFloats.map(f => HWArithmetic.fromDouble(f, 8, 7)), 8, 7)
      val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

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

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      writeCsr(0x00, 1)

      val collected = scala.collection.mutable.ArrayBuffer[Double]()
      var cycles = 0
      val timeout = 20000

      while (collected.length < 2 && cycles < timeout) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          collected += decodeData(dut.io.outStream.stream.payload(0))
        }
        dut.clockDomain.waitSampling()
        cycles += 1
      }

      assert(cycles < timeout, s"8x8 BF16 Band Tiling timed out after $cycles cycles")
      assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

      val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
      assert(dev == 0.0, s"Tiled HW $collected vs SW ${oracle.logits} (max dev = $dev)")
      println(f"[SequentialTest] 8x8 BF16 Band Tiling (tileHeight=4, 2 bands) bit-exact: max dev = $dev%.3f in $cycles cycles")
    }
  }

  test("Sequential: Bias double buffer maintains resident and prefetch state across eager reload (BUG-DDR-01)") {
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3),
      ReLU(),
      Flatten(),
      Linear(inFeatures = 8, outFeatures = 2)
    )

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Accelerator(
        dataType = I8(),
        inputShape = Seq(4, 4, 1),
        modelSpec = spec,
        axiConfig = axiConfig
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memSim.start()

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

      var weightARs = 0L
      dut.clockDomain.onSamplings {
        if (dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean) {
          val addr = dut.io.axiMaster.ar.addr.toLong
          if (addr >= weightBase) weightARs += 1
        }
      }

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

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      // Enable RESIDENT (bit 0) and PREFETCH_EN (bit 1)
      writeCsr(0x10, 3)

      def runPass(pass: Int): Seq[Double] = {
        val inInts = (0 until 16).map(idx => ((idx * 3 + pass * 5) % 7).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

        writeCsr(0x00, 1)

        val collected = scala.collection.mutable.ArrayBuffer[Double]()
        var cycles = 0
        val timeout = 10000

        while (collected.length < 2 && cycles < timeout) {
          if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
            collected += decodeData(dut.io.outStream.stream.payload(0))
          }
          dut.clockDomain.waitSampling()
          cycles += 1
        }

        assert(cycles < timeout, s"Pass $pass timed out after $cycles cycles")
        assert(collected.length == 2, s"Pass $pass: expected 2 outputs, got ${collected.length}")

        val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Pass $pass mismatch: HW $collected vs SW ${oracle.logits}")
        collected.toSeq
      }

      // Pass 0: initial fetch into resident buffer
      val wBefore0 = weightARs
      runPass(0)
      val wDelta0 = weightARs - wBefore0
      assert(wDelta0 > 0, "Initial pass must fetch weights and biases from DDR")

      // Trigger eager reload while idle in prefetch mode
      val wBeforeIdle1 = weightARs
      writeCsr(0x14, 1)
      dut.clockDomain.waitSampling(100)
      val wDeltaIdle1 = weightARs - wBeforeIdle1
      assert(wDeltaIdle1 > 0, "Eager prefetch must fetch weights and biases during idle window")

      // Pass 1: second inference; compute must have zero DDR weight/bias ARs
      val wBeforeActive1 = weightARs
      runPass(1)
      val wDeltaActive1 = weightARs - wBeforeActive1
      assert(wDeltaActive1 == 0L, s"Pass 1: expected 0 weight/bias AR transactions during active compute, got $wDeltaActive1")

      // Trigger next eager reload
      val wBeforeIdle2 = weightARs
      writeCsr(0x14, 1)
      dut.clockDomain.waitSampling(100)
      val wDeltaIdle2 = weightARs - wBeforeIdle2
      assert(wDeltaIdle2 > 0, "Second eager prefetch must fetch during idle window")

      // Pass 2: third inference; compute must have zero DDR weight/bias ARs
      val wBeforeActive2 = weightARs
      runPass(2)
      val wDeltaActive2 = weightARs - wBeforeActive2
      assert(wDeltaActive2 == 0L, s"Pass 2: expected 0 weight/bias AR transactions during active compute, got $wDeltaActive2")

      println("[SequentialTest] BUG-DDR-01: Bias and weight prefetch/resident state verified cleanly across 3 passes")
    }
  }

  test("Arbiter cascade: two-stage tree structure beyond 8 DMA masters") {
    def arbiterCount(nLinears: Int): Int = {
      val targetDir = s"out/test_arb_count_$nLinears"
      val layers = (0 until nLinears).map(_ => Linear(inFeatures = 4, outFeatures = 4))
      // Default idWidth = 8 covers the tree routing budget at these sizes.
      val wideAxi = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 8)
      SpinalConfig(targetDirectory = targetDir).generateVerilog {
        Sequential(
          globalDataType = I8(),
          inputShape = Seq(1, 4),
          layers = layers,
          axiConfig = wideAxi
        )
      }
      val verilog = scala.io.Source.fromFile(s"$targetDir/Sequential.v").mkString
      // Instantiation lines start with the module name (Spinal suffixes
      // per-parameter variants as Axi4ReadOnlyArbiter_2, ...); the module
      // definitions themselves start with the `module` keyword and comments
      // with `//`, so a bare prefix match counts instances only.
      verilog.split("\n").count(_.trim.startsWith("Axi4ReadOnlyArbiter"))
    }

    // 3 linears => 7 triggers <= 8: single-stage, exactly 1 arbiter.
    assert(arbiterCount(3) == 1, "fan-in <= 8 must use a single arbiter")
    // 4 linears => 9 triggers: 2 stage-1 groups + root = 3 arbiters.
    assert(arbiterCount(4) == 3, "9 DMA masters must build a 2+1 tree")
    // 9 linears => 19 triggers: 3 stage-1 groups + root = 4 arbiters.
    assert(arbiterCount(9) == 4, "19 DMA masters must build a 3+1 tree")
  }

  test("Arbiter cascade: bit-exact inference routed through the tree") {
    // 5 linears => 11 triggers => groups of 8+3 under a root arbiter: every
    // weight/bias fetch crosses two arbitration stages with routed IDs.
    val spec = (0 until 5).map(_ => Linear(inFeatures = 4, outFeatures = 4)).toSeq

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Sequential(
        globalDataType = I8(),
        inputShape = Seq(1, 4),
        layers = spec,
        axiConfig = axiConfig
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memory = SparseMemory()
      new Axi4ReadOnlySlaveAgent(dut.io.axiMaster, dut.clockDomain) {
        override def readByte(address: BigInt, id: Int): Byte = memory.read(address.toLong)
      }

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.layers, dut.globalDataType, axiConfig)
      writeWords(memory, weightBase, packed.words)

      val inInts = Seq(1L, -2L, 3L, -4L)
      writeWords(memory, imgBase, MemoryHarness.packBytes(inInts.map(_.toInt)))

      dut.io.start.valid #= false
      dut.io.outStream.stream.ready #= true
      dut.io.imgBaseAddress #= imgBase
      dut.io.weightsBaseAddress #= weightBase
      dut.clockDomain.waitSampling(5)

      dut.io.start.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.start.ready.toBoolean)
      dut.io.start.valid #= false

      val got = scala.collection.mutable.ArrayBuffer[Long]()
      dut.clockDomain.waitSamplingWhere(dut.io.outStream.stream.valid.toBoolean)
      while (got.length < 4) {
        if (dut.io.outStream.stream.valid.toBoolean) {
          dut.io.outStream.stream.payload(0) match {
            case s: SInt => got += s.toBigInt.toLong
            case u: UInt => got += u.toBigInt.toLong
            case b: Bits => got += b.toBigInt.toLong
          }
        }
        dut.clockDomain.waitSampling()
      }

      val oracle = ModelReplica.forwardWithTrace(
        spec, Seq(1, 4), ModelReplica.IntTensor(Seq(1, 4), inInts, 8), packed)
      val expected = oracle.logits
      assert(got.length == expected.length, s"collected ${got.length} beats, expected ${expected.length}")
      for ((g, e) <- got.zip(expected)) {
        assert(g.toDouble == e, s"tree-routed beat $g != oracle $e")
      }
      println("[SequentialTest] Arbiter cascade: 5xLinear bit-exact through the 8+3 tree")
    }
  }

  test("Conv weightLanes: divisor requirement and legacy defaults") {
    // Defaults preserve the historical full widths (bit-identical RTL).
    assert(Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3).effLanes == 9)
    assert(Conv2D(inChannels = 2, outChannels = 3, kernelSize = 3).effLanes == 9)
    assert(Conv1D(inChannels = 2, outChannels = 3, kernelSize = 4).effLanes == 8)
    // Narrow divisors accepted.
    assert(Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, weightLanes = 3).effLanes == 3)
    assert(Conv2D(inChannels = 2, outChannels = 3, kernelSize = 3, weightLanes = 6).effLanes == 6)
    assert(Conv1D(inChannels = 2, outChannels = 3, kernelSize = 4, weightLanes = 2).effLanes == 2)
    // Non-divisors (would need zero-padding) refused like Linear.
    intercept[IllegalArgumentException] { Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, weightLanes = 4) }
    intercept[IllegalArgumentException] { Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, weightLanes = 0) }
    intercept[IllegalArgumentException] { Conv1D(inChannels = 2, outChannels = 3, kernelSize = 4, weightLanes = 3) }
  }

  test("Conv weightLanes: narrow streaming is bit-exact vs replica") {
    // Conv2D K3 streamed 3 lanes at a time (3 chunks of 3): exercises the
    // DMA narrow-beat framing + the matmul K-fold + the replica fold order.
    val spec = Seq(Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3, weightLanes = 3))

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(
      new Sequential(
        globalDataType = BF16(),
        inputShape = Seq(5, 5, 1),
        layers = spec,
        axiConfig = axiConfig
      )
    )

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memory = SparseMemory()
      new Axi4ReadOnlySlaveAgent(dut.io.axiMaster, dut.clockDomain) {
        override def readByte(address: BigInt, id: Int): Byte = memory.read(address.toLong)
      }

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.layers, dut.globalDataType, axiConfig)
      writeWords(memory, weightBase, packed.words)

      val (eW, mW) = (8, 7) // BF16
      val inputValues = (0 until 25).map { idx =>
        val sign = if (idx % 2 == 0) 1.0f else -1.0f
        val mag = (((idx % 7) + 1) * 0.125).toFloat
        HWArithmetic.fromDouble(sign * mag, eW, mW)
      }
      // BF16 elements are 16 bits: pack as padded float words (packBytes is
      // for 8-bit elements only) — same convention as the CLI test scaffold.
      val inFloats = inputValues.map(f => HWArithmetic.decode(f, eW, mW).toFloat)
      writeWords(memory, imgBase, MemoryHarness.packFloats(MemoryHarness.padded(inFloats)))

      dut.io.start.valid #= false
      dut.io.outStream.stream.ready #= true
      dut.io.imgBaseAddress #= imgBase
      dut.io.weightsBaseAddress #= weightBase
      dut.clockDomain.waitSampling(5)

      dut.io.start.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.start.ready.toBoolean)
      dut.io.start.valid #= false

      // Output [3, 3, 2] at lanes=1 => 18 beats.
      val got = scala.collection.mutable.ArrayBuffer[Double]()
      dut.clockDomain.waitSamplingWhere(dut.io.outStream.stream.valid.toBoolean)
      while (got.length < 18) {
        if (dut.io.outStream.stream.valid.toBoolean) {
          got += decodeData(dut.io.outStream.stream.payload(0))
        }
        dut.clockDomain.waitSampling()
      }

      val oracle = ModelReplica.forwardWithTrace(
        spec, Seq(5, 5, 1), ModelReplica.FloatTensor(Seq(5, 5, 1), inputValues, eW, mW), packed)
      val expected = oracle.logits
      assert(got.length == expected.length, s"collected ${got.length} beats, expected ${expected.length}")
      for (((g, e), i) <- got.zip(expected).zipWithIndex) {
        assert(g == e, s"narrow-lane beat $i: hw=$g != replica=$e")
      }
      println("[SequentialTest] Conv weightLanes=3: bit-exact vs replica (K-fold order)")
    }
  }

  test("BUG-DDR-06: idWidth validation and elaboration for deep models (> 8 weight/bias layers)") {
    // 9 linear layers => 18 DMA triggers for weights/biases + 1 for image = 19 triggers.
    // log2Up(19) = 5 bits.
    val layers = (0 until 9).map { _ =>
      Linear(inFeatures = 4, outFeatures = 4)
    }

    // 1. If axiConfig.idWidth is too small (< 5 bits), an explicit and informative exception is thrown
    val ex = intercept[IllegalArgumentException] {
      SpinalConfig().generateVerilog {
        Sequential(
          globalDataType = I8(),
          inputShape = Seq(1, 4),
          layers = layers,
          axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
        )
      }
    }
    assert(ex.getMessage.contains("insufficient to arbitrate 19 DMA masters"))

    // 2. With adequate idWidth (>= 5 bits), elaboration succeeds cleanly
    SpinalConfig().generateVerilog {
      Sequential(
        globalDataType = I8(),
        inputShape = Seq(1, 4),
        layers = layers,
        axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 5)
      )
    }

    // 3. With default axiConfig (idWidth = 8), deep models elaborate without manual configuration
    SpinalConfig().generateVerilog {
      Sequential(
        globalDataType = I8(),
        inputShape = Seq(1, 4),
        layers = layers
      )
    }
  }
}
