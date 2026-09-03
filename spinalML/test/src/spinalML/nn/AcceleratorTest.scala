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

      // 1. Prepare deterministic weights and input data
      val packed = WeightMemoryLayout.buildDeterministicWeights(dut.modelSpec, dut.globalDataType, axiConfig)
      writeWords(memSim.memory, weightBase, packed.words)

      val inInts = (0 until 16).map(idx => (idx % 3).toLong)
      val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      writeWords(memSim.memory, imgBase, imgWords)

      // Calculate oracle
      val inputTensor = ModelReplica.IntTensor(Seq(4, 4, 1), inInts, 8)
      val oracle = ModelReplica.forwardWithTrace(dut.modelSpec, dut.inputShape, inputTensor, packed)

      // 2. AXI-Lite helpers
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

      // Initialize control bus
      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      dut.clockDomain.waitSampling(5)

      // Program addresses
      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

      // Pulse START
      writeCsr(0x00, 1)

      // Collect outputs
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

      // Set bases
      writeCsr(0x08, imgBase)
      writeCsr(0x0C, weightBase)

      // Enable RUN mode (CSR 0x1C bit 0 = 1)
      writeCsr(0x1C, 1)

      // Single START pulse to initiate continuous auto-advance
      writeCsr(0x00, 1)

      // Monitor continuous frames with bounded loop
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
}
