// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.I8
import spinalML.harness.MemoryHarness
import spinalML.replica.{ModelReplica, WeightMemoryLayout}

class AcceleratorTest extends AnyFunSuite {
  /** Accelerator SoC suite on the compact 4x4 toy model: CSR control, single
   * and continuous streaming, DDR write-back, residency/reload and eager
   * prefetch — bit-exact vs ModelReplica, AR traffic-metered. */
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)

  val imgBase = 0x10000L
  val weightBase = 0x20000L

  private def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) {
      mem.writeBigInt(base + i * 8, w, 8)
    }
  }

  private def decodeData(p: Data): Double = {
    p match {
      case s: SInt => s.toBigInt.toLong.toDouble
      case u: UInt => u.toBigInt.toLong.toDouble
      case b: Bits => b.toBigInt.toLong.toDouble
      case bt: BaseType => bt.toBigInt.toDouble
      case _ => throw new IllegalArgumentException(s"Unsupported data type: $p")
    }
  }

  // Compact 4x4 image accelerator for fast, deterministic SoC verification
  private def makeToyAccelerator(tileHeight: Int = -1): Accelerator[Data] = {
    val spec = Seq(
      Conv2D(inChannels = 1, outChannels = 2, kernelSize = 3),
      ReLU(),
      Flatten(),
      Linear(inFeatures = 8, outFeatures = 2)
    )
    new Accelerator(
      dataType = I8(),
      inputShape = Seq(4, 4, 1),
      modelSpec = spec,
      axiConfig = axiConfig,
      tileHeight = tileHeight
    )
  }

  test("Accelerator: Single inference CSR control (start, status busy/done, bit-exact outputs)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      val inInts = (0 until 16).map(idx => (idx % 3).toLong)
      val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      writeWords(memSim.memory, imgBase, imgWords)

      val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
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
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

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

      assert(cycles < timeout, "Inference timed out")
      assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

      val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
      assert(dev == 0.0, s"Bit-exact mismatch: HW $collected vs SW ${oracle.logits}")
      println(f"[AcceleratorTest] Single inference bit-exact: max dev = $dev%.3f in $cycles cycles")
    }
  }

  test("Accelerator: Continuous streaming execution (CSR 0x1C RUN, auto-advance, TILE_CNT, STOP)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      val numFrames = 4
      val imageBytes = 16 // 4x4 bytes
      val oracles = scala.collection.mutable.ArrayBuffer[Seq[Double]]()

      for (k <- 0 until numFrames) {
        val inInts = (0 until 16).map(idx => ((idx + k) % 3).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase + k * imageBytes, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)
        oracles += oracle.logits
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
      dut.clockDomain.waitSampling(5)

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

      // Enable RUN mode (CSR 0x1C bit 0 = 1)
      writeCsr(0x1C, 1)

      // Single START pulse to initiate continuous auto-advance
      writeCsr(0x00, 1)

      val allOutputs = scala.collection.mutable.ArrayBuffer[Seq[Double]]()
      val currentFrameOutputs = scala.collection.mutable.ArrayBuffer[Double]()
      var silence = 0
      var timeout = 0
      val maxTimeout = 20000
      var stopIssued = false

      while (silence < 2000 && timeout < maxTimeout && allOutputs.length < numFrames) {
        timeout += 1
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          silence = 0
          currentFrameOutputs += decodeData(dut.io.outStream.stream.payload(0))
          if (currentFrameOutputs.length == 2) {
            val k = allOutputs.length
            allOutputs += currentFrameOutputs.toVector
            currentFrameOutputs.clear()
            println(s"[AcceleratorTest] Frame $k completed")

            if (!stopIssued && allOutputs.length == 2) {
              println(s"[AcceleratorTest] Issuing STOP after frame $k...")
              writeCsr(0x1C, 0)
              stopIssued = true
              println(s"[AcceleratorTest] STOP issued successfully")
            }
          }
        } else {
          silence += 1
        }
        dut.clockDomain.waitSampling()
      }

      val expectedFrames = 3
      assert(stopIssued, "STOP was never issued")
      assert(allOutputs.length == expectedFrames, s"Expected $expectedFrames frames after in-flight STOP, collected ${allOutputs.length}")

      for (k <- 0 until expectedFrames) {
        val dev = allOutputs(k).zip(oracles(k)).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Frame $k mismatch: HW ${allOutputs(k)} vs SW ${oracles(k)}")
      }

      // Check TILE_CNT (0x18)
      val tileCnt = readCsr(0x18)
      assert(tileCnt == expectedFrames, s"Expected TILE_CNT == $expectedFrames, got $tileCnt")

      // Sample quiet cycles after STOP to verify engine stays stopped
      dut.clockDomain.waitSampling(50)
      assert(!dut.io.outStream.stream.valid.toBoolean, "Engine continued producing data after STOP")

      println(s"[AcceleratorTest] Continuous streaming PASSED ($numFrames frames bit-exact, clean STOP confirmed)")
    }
  }

  test("Accelerator: Write-back to DDR via DMAWriter (CSR 0x20 OUT_ADDR, CSR 0x24 OUT_CTRL, bit-exact DDR check)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      val inInts = (0 until 16).map(idx => (idx % 3).toLong)
      val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      writeWords(memSim.memory, imgBase, imgWords)

      val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
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
      dut.io.outStream.stream.ready #= false // Consumer ready=0 to prove write-back doesn't stall on outStream
      dut.clockDomain.waitSampling(5)

      val outBase = 0x30000L

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

      // Configure DMA write-back
      writeCsr(0x20, outBase)
      writeCsr(0x24, 1) // writeToDdr = true

      // Verify DMA status is idle before start
      val dmaStatusBefore = readCsr(0x28)
      assert(dmaStatusBefore == 0, s"DMA writer should be idle before start, got 0x$dmaStatusBefore%X")

      writeCsr(0x00, 1)

      var cycles = 0
      val timeout = 10000
      var doneObserved = false

      while (cycles < timeout && (!doneObserved || dut.io.busy.toBoolean)) {
        if (dut.io.done.toBoolean) {
          doneObserved = true
        }
        // In writeToDdr mode, outStream.stream.valid must stay false
        assert(!dut.io.outStream.stream.valid.toBoolean, "outStream.valid pulsed high while writeToDdr was active!")
        dut.clockDomain.waitSampling()
        cycles += 1
      }

      assert(cycles < timeout, "Inference and DDR write-back timed out")
      assert(doneObserved, "Done pulse was never observed")
      assert(!dut.io.busy.toBoolean, "Accelerator stayed busy after completion")

      // Check DDR memory contents at outBase
      // The toy model produces 2 elements of I8 (outFeatures = 2)
      // On a 64-bit bus, this is in byte 0 and byte 1 of the 64-bit word at outBase
      val writtenWord = memSim.memory.readBigInt(outBase, 8)
      val hwLogits = (0 until 2).map { b =>
        val byteVal = ((writtenWord >> (b * 8)) & 0xFF).toLong
        val signedByte = if (byteVal >= 128) byteVal - 256 else byteVal
        signedByte.toDouble
      }

      val dev = hwLogits.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
      assert(dev == 0.0, s"DDR Write-Back Bit-Exact mismatch: HW $hwLogits vs SW ${oracle.logits}")
      println(f"[AcceleratorTest] DDR Write-Back bit-exact: max dev = $dev%.3f in $cycles cycles (HW=$hwLogits, SW=${oracle.logits})")
    }
  }

  test("Accelerator: Continuous streaming write-back to DDR (CSR 0x1C RUN + CSR 0x24 OUT_CTRL, auto-increment OUT_ADDR)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      val numFrames = 3
      val imageBytes = 16 // 4x4 bytes
      val oracles = scala.collection.mutable.ArrayBuffer[Seq[Double]]()

      for (k <- 0 until numFrames) {
        val inInts = (0 until 16).map(idx => ((idx + k) % 3).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase + k * imageBytes, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)
        oracles += oracle.logits
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
      dut.io.outStream.stream.ready #= false
      dut.clockDomain.waitSampling(5)

      val outBase = 0x30000L
      val outFrameBytes = 8 // 1 beat on 64-bit bus for 2 I8 elements

      // Pre-fill output region with poison pattern 0xAA to detect untouched memory
      for (k <- 0 until numFrames) {
        memSim.memory.writeBigInt(outBase + k * outFrameBytes, BigInt("AAAAAAAAAAAAAAAA", 16), 8)
      }

      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)
      writeCsr(0x20, outBase)
      writeCsr(0x24, 1) // writeToDdr = true
      writeCsr(0x1C, 1) // RUN = true

      writeCsr(0x00, 1)

      var timeout = 0
      val maxTimeout = 20000
      var stopIssued = false
      var completedFrames = 0

      while (timeout < maxTimeout && completedFrames < numFrames) {
        timeout += 1
        if (dut.io.done.toBoolean) {
          completedFrames += 1
          println(s"[AcceleratorTest] Frame done pulse observed: count=$completedFrames")
          if (!stopIssued && completedFrames == 2) {
            println(s"[AcceleratorTest] Issuing STOP after 2 frames...")
            writeCsr(0x1C, 0)
            stopIssued = true
          }
        }
        dut.clockDomain.waitSampling()
      }

      assert(stopIssued, "STOP was never issued")
      assert(completedFrames == numFrames, s"Expected $numFrames frames, observed $completedFrames")

      var idleCycles = 0
      while (dut.io.busy.toBoolean && idleCycles < 200) {
        dut.clockDomain.waitSampling()
        idleCycles += 1
      }
      assert(!dut.io.busy.toBoolean, "Accelerator stayed busy")

      val tileCnt = readCsr(0x18)
      assert(tileCnt == numFrames, s"Expected TILE_CNT == $numFrames, got $tileCnt")

      // Verify that EACH frame was written to its respective address without overwriting
      for (k <- 0 until numFrames) {
        val frameAddr = outBase + k * outFrameBytes
        val writtenWord = memSim.memory.readBigInt(frameAddr, 8)
        val hwLogits = (0 until 2).map { b =>
          val byteVal = ((writtenWord >> (b * 8)) & 0xFF).toLong
          val signedByte = if (byteVal >= 128) byteVal - 256 else byteVal
          signedByte.toDouble
        }
        val dev = hwLogits.zip(oracles(k)).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, f"Frame $k at 0x$frameAddr%X mismatch: HW $hwLogits vs SW ${oracles(k)}")
      }

      println(s"[AcceleratorTest] Continuous streaming DDR write-back PASSED ($numFrames frames bit-exact at distinct addresses)")
    }
  }

  test("Accelerator: Weight residency (CSR 0x10 WEIGHT_RESIDENT, zero DDR weight traffic in steady state, CSR 0x14 RELOAD)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      var arWeightCount = 0L
      var arImgCount = 0L
      dut.clockDomain.onSamplings {
        if (dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean) {
          val a = dut.io.axiMaster.ar.payload.addr.toLong
          if (a >= weightBase) arWeightCount += 1
          else if (a >= imgBase) arImgCount += 1
        }
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

      def runInference(imgId: Int): Seq[Double] = {
        val inInts = (0 until 16).map(idx => ((idx + imgId) % 5).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

        val collected = scala.collection.mutable.ArrayBuffer[Double]()
        writeCsr(0x00, 1)

        var cycles = 0
        val timeout = 5000
        while (collected.length < 2 && cycles < timeout) {
          if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
            collected += decodeData(dut.io.outStream.stream.payload(0))
          }
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(cycles < timeout, "Inference timed out")
        assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

        val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Mismatch for img $imgId: HW $collected vs SW ${oracle.logits}")

        while (dut.io.busy.toBoolean) dut.clockDomain.waitSampling()
        dut.clockDomain.waitSampling(5)

        collected.toSeq
      }

      // Pass 0: Baseline streaming (CSR 0x10 = 0)
      val snap0Weight = arWeightCount
      runInference(0)
      val p0WeightTraffic = arWeightCount - snap0Weight
      assert(p0WeightTraffic > 0, "Pass 0 baseline should fetch weights from DDR")

      // Enable WEIGHT_RESIDENT (CSR 0x10 bit 0 = 1)
      writeCsr(0x10, 1)

      // Pass 1: Rising edge of resident mode triggers initial resident load
      val snap1Weight = arWeightCount
      runInference(1)
      val p1WeightTraffic = arWeightCount - snap1Weight
      assert(p1WeightTraffic > 0, "Pass 1 should fetch initial resident weights")

      // Pass 2: Steady resident state -> ZERO weight AR traffic expected
      val snap2Weight = arWeightCount
      runInference(2)
      val p2WeightTraffic = arWeightCount - snap2Weight
      assert(p2WeightTraffic == 0, s"Pass 2 should have ZERO weight DDR traffic in resident mode, got $p2WeightTraffic")

      // Pass 3: Steady resident state -> ZERO weight AR traffic expected
      val snap3Weight = arWeightCount
      runInference(3)
      val p3WeightTraffic = arWeightCount - snap3Weight
      assert(p3WeightTraffic == 0, s"Pass 3 should have ZERO weight DDR traffic in resident mode, got $p3WeightTraffic")

      // RELOAD: write to CSR 0x14 arms a single refetch
      writeCsr(0x14, 1)

      // Pass 4: Next START executes the reload -> weight AR traffic expected
      val snap4Weight = arWeightCount
      runInference(4)
      val p4WeightTraffic = arWeightCount - snap4Weight
      assert(p4WeightTraffic > 0, "Pass 4 should refetch weights following RELOAD")

      // Pass 5: Back to steady resident state -> ZERO weight AR traffic
      val snap5Weight = arWeightCount
      runInference(5)
      val p5WeightTraffic = arWeightCount - snap5Weight
      assert(p5WeightTraffic == 0, s"Pass 5 should resume ZERO weight traffic, got $p5WeightTraffic")

      println("[AcceleratorTest] Weight residency & reload PASSED (steady state zero-traffic strictly confirmed)")
    }
  }

  test("Accelerator: Weight eager prefetch (CSR 0x10 PREFETCH_EN, background IDLE bank fill, zero START latency)") {
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator())

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

      var arWeightCount = 0L
      dut.clockDomain.onSamplings {
        if (dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean) {
          val a = dut.io.axiMaster.ar.payload.addr.toLong
          if (a >= weightBase) arWeightCount += 1
        }
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

      def runInference(imgId: Int): Seq[Double] = {
        val inInts = (0 until 16).map(idx => ((idx + imgId) % 5).toLong)
        val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        writeWords(memSim.memory, imgBase, imgWords)

        val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
        val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

        val collected = scala.collection.mutable.ArrayBuffer[Double]()
        writeCsr(0x00, 1)

        var cycles = 0
        val timeout = 5000
        while (collected.length < 2 && cycles < timeout) {
          if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
            collected += decodeData(dut.io.outStream.stream.payload(0))
          }
          dut.clockDomain.waitSampling()
          cycles += 1
        }
        assert(cycles < timeout, "Inference timed out")
        assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

        val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Mismatch for img $imgId: HW $collected vs SW ${oracle.logits}")

        while (dut.io.busy.toBoolean) dut.clockDomain.waitSampling()
        dut.clockDomain.waitSampling(5)

        collected.toSeq
      }

      // Step 1: Baseline inference in resident mode (CSR 0x10 = 1) to establish resident bank
      writeCsr(0x10, 1) // WEIGHT_RESIDENT = 1, PREFETCH_EN = 0
      runInference(0)   // Initial resident load into Ping bank

      // Step 2: Enable Eager Prefetch mode (CSR 0x10 = 3) and trigger background RELOAD
      writeCsr(0x10, 3) // WEIGHT_RESIDENT = 1, PREFETCH_EN = 1
      val snapBeforeReload = arWeightCount
      writeCsr(0x14, 1) // RELOAD: triggers eager prefetch immediately into IDLE (Pong) bank

      // Wait for eager background prefetch to complete while accelerator is IDLE
      var totalWait = 0
      var quietCycles = 0
      var lastWeightCount = arWeightCount
      while (quietCycles < 150 && totalWait < 3000) {
        dut.clockDomain.waitSampling()
        totalWait += 1
        if (arWeightCount != lastWeightCount) {
          lastWeightCount = arWeightCount
          quietCycles = 0
        } else if (arWeightCount > snapBeforeReload) {
          quietCycles += 1
        }
      }
      assert(arWeightCount > snapBeforeReload, "Eager prefetch should fetch weights immediately into IDLE bank before START")

      val snapAfterPrefetch = arWeightCount
      println(s"[AcceleratorTest] Eager prefetch completed in $totalWait cycles (total weight ARs = $snapAfterPrefetch)")

      // Step 3: Run inference 1 with prefetched weights
      // Because weights were already prefetched eagerly, DURING inference there should be ZERO weight traffic!
      runInference(1)
      val duringInferenceTraffic = arWeightCount - snapAfterPrefetch
      assert(duringInferenceTraffic == 0, s"During inference, weight traffic should be ZERO under eager prefetch, got $duringInferenceTraffic")

      // Step 4: Subsequent steady state inference 2 confirms the swapped bank remains resident
      val snapSteady = arWeightCount
      runInference(2)
      assert(arWeightCount - snapSteady == 0, "Post-swap steady state should have ZERO weight traffic")

      println("[AcceleratorTest] Weight eager prefetch PASSED (eager background fill & governed swap verified)")
    }
  }

  test("Accelerator: Vertical band tiling (tileHeight = 2, multi-band seam continuity and bit-exact outputs)") {
    // Compile banded model with tileHeight = 2 (2 bands of 2x4 = 8 elements each)
    val compiledBanded = SimConfig.withVerilator.withConfig(spinalConfig).compile(makeToyAccelerator(tileHeight = 2))

    compiledBanded.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val memSim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8)
      )
      memSim.start()

      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

      val inInts = (0 until 16).map(idx => (idx % 3).toLong)
      val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      writeWords(memSim.memory, imgBase, imgWords)

      // Expected output calculated via ModelReplica oracle
      val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
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
      val timeout = 10000

      while (collected.length < 2 && cycles < timeout) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          collected += decodeData(dut.io.outStream.stream.payload(0))
        }
        dut.clockDomain.waitSampling()
        cycles += 1
      }

      assert(cycles < timeout, "Banded inference timed out")
      assert(collected.length == 2, s"Expected 2 outputs, got ${collected.length}")

      val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
      assert(dev == 0.0, s"Banded inference bit-exact mismatch: HW $collected vs SW ${oracle.logits}")
      println(f"[AcceleratorTest] Vertical band tiling (tileHeight=2) PASSED: bit-exact with oracle in $cycles cycles")
    }
  }

  test("Generate Verilog for Python co-simulation") {
    // Identity passthrough: exercises the AXI-Lite control plane, the output
    // stream, continuous RUN/STOP and the DMAWriter write-back path without
    // needing a weight-layout oracle on the Python side.
    SpinalConfig().generateVerilog {
      val dut = new Accelerator(
        dataType = I8(),
        inputShape = Seq(4, 4, 1),
        modelSpec = Seq(Flatten()),
        axiConfig = axiConfig
      )
      dut.setDefinitionName("AcceleratorPassthroughTestComp")
      dut
    }

    // Runtime dequantization boundary: CSR 0x30 programs the Cast scale
    // (mirrors examples/Mnist/Model.scala + inference.py).
    SpinalConfig().generateVerilog {
      val dut = new Accelerator(
        dataType = I8(),
        inputShape = Seq(4, 4, 1),
        modelSpec = Seq(Flatten(), Cast(spinalML.dtypes.BF16(), runtimeScale = true)),
        axiConfig = axiConfig
      )
      dut.setDefinitionName("AcceleratorRuntimeCastTestComp")
      dut
    }
  }
}
