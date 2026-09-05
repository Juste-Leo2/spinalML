package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}
import spinalML.dtypes.{I8, I32}
import spinalML.harness.MemoryHarness
import spinalML.nn.{Accelerator, Conv2D, Flatten, Linear, ReLU}
import spinalML.replica.{ModelReplica, WeightMemoryLayout}

class StreamDoubleBufferTest extends AnyFunSuite {
  test("Double Buffering Ping-Pong logic") {
    SimConfig.compile {
      val dut = StreamDoubleBuffer(I32(), depth = 8, lanes = 2) // 4 addresses per bank
      dut.setDefinitionName("StreamDoubleBufferTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      
      dut.io.streamIn.valid #= false
      dut.io.nextTile #= false
      dut.io.reArm #= false
      dut.io.readAddr #= 0
      dut.clockDomain.waitSampling()
      
      // Load Ping bank (4 writes of 2 lanes)
      for (i <- 0 until 4) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= i * 2
        dut.io.streamIn.payload(1) #= i * 2 + 1
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      dut.clockDomain.waitSampling() // Wait for pingFull register to update
      
      assert(dut.io.tileReady.toBoolean == true, "Ping bank should be ready")
      
      // Read from Ping bank
      for (i <- 0 until 4) {
        dut.io.readAddr #= i
        dut.clockDomain.waitSampling() // wait for readAddr to register
        dut.clockDomain.waitSampling() // latency
        assert(dut.io.readData(0).toInt == i * 2)
        assert(dut.io.readData(1).toInt == i * 2 + 1)
      }
      
      // While reading Ping, load Pong
      for (i <- 0 until 4) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= 100 + i * 2
        dut.io.streamIn.payload(1) #= 100 + i * 2 + 1
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      
      // Finish Ping, switch to Pong
      dut.io.nextTile #= true
      dut.clockDomain.waitSampling()
      dut.io.nextTile #= false
      
      // Read from Pong
      dut.io.readAddr #= 0
      dut.clockDomain.waitSampling()
      dut.clockDomain.waitSampling() // latency
      assert(dut.io.readData(0).toInt == 100)
    }
  }

  test("StreamDoubleBuffer: Component-level weight residency (residentHold) and prefetch staging") {
    SimConfig.compile {
      val dut = StreamDoubleBuffer(I32(), depth = 4, lanes = 1, enableFreezePort = true)
      dut.setDefinitionName("StreamDoubleBufferResidencyComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      dut.io.streamIn.valid #= false
      dut.io.nextTile #= false
      dut.io.reArm #= false
      dut.io.residentHold.foreach(_ #= false)
      dut.io.stageRequest.foreach(_ #= false)
      dut.io.readAddr #= 0
      dut.clockDomain.waitSampling(5)

      // 1. Load initial tile into Ping bank (4 elements: 10, 20, 30, 40)
      for (v <- Seq(10, 20, 30, 40)) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= v
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      dut.clockDomain.waitSampling(2)
      assert(dut.io.tileReady.toBoolean, "Ping bank should be ready")

      // 2. Enable residentHold
      dut.io.residentHold.foreach(_ #= true)

      // Read Ping bank repeatedly across nextTile pulses without losing data
      for (pass <- 0 until 3) {
        for (i <- 0 until 4) {
          dut.io.readAddr #= i
          dut.clockDomain.waitSampling(2)
          assert(dut.io.readData(0).toInt == (i + 1) * 10, s"Pass $pass: expected ${(i + 1) * 10}, got ${dut.io.readData(0).toInt}")
        }
        // Pulse nextTile: residentHold should PREVENT flipping to an empty bank
        dut.io.nextTile #= true
        dut.clockDomain.waitSampling()
        dut.io.nextTile #= false
        dut.clockDomain.waitSampling()
        assert(dut.io.tileReady.toBoolean, s"Pass $pass: Ping bank should remain held and ready under residentHold")
      }

      // 3. Prefetch staging: while Ping is held, stage a fresh tile into Pong
      dut.io.stageRequest.foreach(_ #= true)
      for (v <- Seq(100, 200, 300, 400)) {
        dut.io.streamIn.valid #= true
        dut.io.streamIn.payload(0) #= v
        dut.clockDomain.waitSampling()
      }
      dut.io.streamIn.valid #= false
      dut.clockDomain.waitSampling(2)

      // Pulse nextTile: now that a staged fresh tile is loaded, nextTile flips computeBank onto Pong!
      dut.io.nextTile #= true
      dut.clockDomain.waitSampling()
      dut.io.nextTile #= false
      dut.clockDomain.waitSampling(2)

      // Read fresh Pong data
      for (i <- 0 until 4) {
        dut.io.readAddr #= i
        dut.clockDomain.waitSampling(2)
        assert(dut.io.readData(0).toInt == (i + 1) * 100, s"Pong staged read: expected ${(i + 1) * 100}, got ${dut.io.readData(0).toInt}")
      }
      println("[StreamDoubleBufferTest] Component-level residency hold and prefetch staging PASSED")
    }
  }

  test("StreamDoubleBuffer: System-level weight residency keeps outputs bit-exact with zero weight DDR traffic") {
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val imgBase = 0x10000L
    val weightBase = 0x20000L

    def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
      for ((w, i) <- words.zipWithIndex) {
        mem.writeBigInt(base + i * 8, w, 8)
      }
    }

    def decodeData(p: Data): Double = {
      p match {
        case s: SInt => s.toBigInt.toLong.toDouble
        case u: UInt => u.toBigInt.toLong.toDouble
        case b: Bits => b.toBigInt.toLong.toDouble
        case bt: BaseType => bt.toBigInt.toDouble
        case _ => throw new IllegalArgumentException(s"Unsupported data type: $p")
      }
    }

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
        axiConfig = axiConfig,
        weightResidencyCSR = true
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

      // Meter AR transactions
      var imgARs = 0L
      var weightARs = 0L
      dut.clockDomain.onSamplings {
        if (dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean) {
          val addr = dut.io.axiMaster.ar.addr.toLong
          if (addr >= weightBase) weightARs += 1
          else if (addr >= imgBase) imgARs += 1
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
      writeCsr(0x10, 0) // baseline STREAM_PER_PASS

      def runInference(passId: Int): Seq[Double] = {
        val inInts = (0 until 16).map(idx => ((idx * 2 + passId * 3) % 5).toLong)
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

        assert(cycles < timeout, s"Pass $passId timed out")
        assert(collected.length == 2, s"Pass $passId: expected 2 outputs, got ${collected.length}")

        val dev = collected.zip(oracle.logits).map { case (hw, sw) => math.abs(hw - sw) }.max
        assert(dev == 0.0, s"Pass $passId mismatch: HW $collected vs SW ${oracle.logits}")
        collected.toSeq
      }

      // Pass 0: Baseline — weights MUST be fetched from DDR
      val wBefore0 = weightARs
      runInference(0)
      val wDelta0 = weightARs - wBefore0
      assert(wDelta0 > 0, s"Baseline pass 0 should have read weights from DDR, got $wDelta0 ARs")

      // Pass 1: Switch to WEIGHT_RESIDENT (CSR 0x10 = 1)
      // The mode transition rising edge self-fetches weights
      writeCsr(0x10, 1)
      runInference(1)

      // Passes 2 and 3: Steady-state resident — weight DDR traffic MUST be strictly ZERO!
      for (p <- 2 to 3) {
        val wBefore = weightARs
        runInference(p)
        val wDelta = weightARs - wBefore
        assert(wDelta == 0L, s"Pass $p: expected 0 weight AR transactions in resident mode, got $wDelta")
        println(f"[StreamDoubleBufferTest] Pass $p resident: weight ARs = $wDelta (zero DDR traffic verified)")
      }

      // Pass 4: Issue one-shot RELOAD trigger (CSR 0x14 = 1)
      writeCsr(0x14, 1)
      val wBeforeReload = weightARs
      runInference(4)
      val wDeltaReload = weightARs - wBeforeReload
      assert(wDeltaReload > 0, s"Pass 4 after RELOAD should have re-fetched weights, got $wDeltaReload ARs")
      println(f"[StreamDoubleBufferTest] Pass 4 after RELOAD: weight ARs = $wDeltaReload (refetch verified)")

      // Pass 5: Residency resumes — zero weight ARs again
      val wBefore5 = weightARs
      runInference(5)
      val wDelta5 = weightARs - wBefore5
      assert(wDelta5 == 0L, s"Pass 5: expected 0 weight AR transactions in resumed residency, got $wDelta5")
      println(f"[StreamDoubleBufferTest] Pass 5 resident: weight ARs = $wDelta5 (zero DDR traffic resumed)")

      // ---- Eager Prefetch Mode (CSR 0x10 = 3: RESIDENT + PREFETCH_EN) ----
      // Pass 6: Switch to RESIDENT + PREFETCH mode and issue RELOAD
      writeCsr(0x10, 3)
      val wBeforeIdle6 = weightARs
      writeCsr(0x14, 1) // Issue RELOAD while accelerator is idle (pre-START)
      dut.clockDomain.waitSampling(200) // Settle: allow eager fetch to land into idle bank
      val wDeltaIdle6 = weightARs - wBeforeIdle6
      assert(wDeltaIdle6 > 0, s"Pass 6: eager fetch should occur during idle window BEFORE start, got $wDeltaIdle6 ARs")
      println(f"[StreamDoubleBufferTest] Pass 6 prefetch: eager idle weight ARs = $wDeltaIdle6 (pre-loaded before START)")

      // Active inference: weights are already prefetched, so weight ARs during active compute must be STRICTLY ZERO!
      val wBeforeActive6 = weightARs
      runInference(6)
      val wDeltaActive6 = weightARs - wBeforeActive6
      assert(wDeltaActive6 == 0L, s"Pass 6: expected 0 weight AR transactions during active inference under prefetch, got $wDeltaActive6")
      println(f"[StreamDoubleBufferTest] Pass 6 inference: active weight ARs = $wDeltaActive6 (zero DDR traffic during compute)")

      // Pass 7: Eager prefetch from an alternate address (weightBaseB) to confirm genuine refetch
      val weightBaseB = 0x40000L
      writeWords(memSim.memory, weightBaseB, packed.words)
      writeCsr(0x0C, weightBaseB) // Update weights base address
      val wBeforeIdle7 = weightARs
      writeCsr(0x14, 1) // Reload from new address while idle
      dut.clockDomain.waitSampling(200)
      val wDeltaIdle7 = weightARs - wBeforeIdle7
      assert(wDeltaIdle7 > 0, s"Pass 7: eager fetch from new base address should occur during idle, got $wDeltaIdle7 ARs")
      println(f"[StreamDoubleBufferTest] Pass 7 prefetch: eager idle weight ARs from alternate base = $wDeltaIdle7")

      val wBeforeActive7 = weightARs
      runInference(7)
      val wDeltaActive7 = weightARs - wBeforeActive7
      assert(wDeltaActive7 == 0L, s"Pass 7: expected 0 weight AR transactions during active inference, got $wDeltaActive7")
      println(f"[StreamDoubleBufferTest] Pass 7 inference: active weight ARs = $wDeltaActive7 (zero DDR traffic during compute)")

      // Pass 8: Return to standard resident mode (CSR 0x10 = 1) with serialized reload
      writeCsr(0x10, 1)
      writeCsr(0x0C, weightBase)
      writeCsr(0x14, 1)
      val wBefore8 = weightARs
      runInference(8)
      val wDelta8 = weightARs - wBefore8
      assert(wDelta8 > 0, s"Pass 8 after return to resident mode with reload: expected >0 ARs, got $wDelta8")
      println(f"[StreamDoubleBufferTest] Pass 8 resident reload: weight ARs = $wDelta8")

      // Pass 9: Steady resident state resumes — zero weight ARs verified
      val wBefore9 = weightARs
      runInference(9)
      val wDelta9 = weightARs - wBefore9
      assert(wDelta9 == 0L, s"Pass 9: expected 0 weight AR transactions in resumed residency, got $wDelta9")
      println(f"[StreamDoubleBufferTest] Pass 9 steady resident: weight ARs = $wDelta9 (zero DDR traffic resumed)")

      println("[StreamDoubleBufferTest] All residency, reload and eager prefetch contract checks PASSED cleanly!")
    }
  }

  test("Generate Verilog for Python Cocotb") {
    SpinalConfig().generateVerilog {
      val dut = StreamDoubleBuffer(I32(), depth = 8, lanes = 2) // 4 addresses per bank
      dut.setDefinitionName("StreamDoubleBufferTestComp")
      dut
    }
  }
}
