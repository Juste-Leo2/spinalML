// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinalML.dtypes.I8
import spinalML.harness.MemoryHarness
import spinalML.replica.{ModelReplica, WeightMemoryLayout}

/**
 * Scale-spill hang debug (minimal repro, NO endless wait).
 *
 * Bisect rappel: 1-beat spill passe, 2-beats bloque sur le moteur
 * Linear/MatmulOp (les suites existantes ne couvrent que du 1-beat en
 * Sequential complet: M*N=4 I8 => 1 beat; le 2-beats ne passait qu'en
 * wrapper isole SpillPassLoopWrapper, sans moteur).
 *
 * - "control-1beat": Linear(8,8) Ks=4 => M*N=8 I8 => 1 beat, doit passer.
 * - "repro-2beats": Linear(16,16) Ks=8 => M*N=16 I8 => 2 beats, hang suspecte.
 *
 * Discipline anti-interminable:
 * - timeout dur 20000 cycles par run (assert + dump, jamais de while infini);
 * - echantillonnage espace: sondes simPublic du controleur + bus lues tous
 *   les 100 cycles seulement (pas de lecture DUT par cycle sauf le valid de
 *   sortie); progression loggee tous les 2000 cycles;
 * - sortie fichier via CLI: `test` redirige vers out/scale-debug.log.
 */
class ScaleSpillDebugTest extends AnyFunSuite {

  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val imgBase = 0x10000L
  val weightBase = 0x40000L
  val spillBase = 0x30000L

  private def writeWords(mem: spinal.lib.bus.amba4.axi.sim.SparseMemory, base: Long, words: Seq[BigInt]): Unit = {
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)
  }

  private def runCase(spec: Seq[LayerSpec], inShape: Seq[Int], label: String): Unit = {
    var packedWords: Seq[BigInt] = null
    var imgWords: Seq[BigInt] = null
    var expected: Seq[Double] = null
    var weightBytesTotal = 0

    val compiled = SimConfig.withWave.compile({
      new Accelerator(
        dataType = I8(),
        inputShape = inShape,
        modelSpec = spec,
        axiConfig = axiConfig,
        temporal = 1,
        memory = MemorySpec(spillBase = Some(spillBase), capacityBytes = Some(65536))
      )
    })

    SpinalConfig(targetDirectory = "out/tmp-scale-spill-debug").generateVerilog(new Component {
      setDefinitionName("ScaleSpillDebugLayout")
      val packed = WeightMemoryLayout.buildDeterministicWeights(spec, I8(), axiConfig)
      packedWords = packed.words
      weightBytesTotal = packed.totalBytes
      val k = inShape.product
      val inInts = (0 until k).map(idx => (((idx * 7 + 3) % 15) - 7).toLong)
      imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
      expected = ModelReplica.forwardWithTrace(
        spec, inShape, ModelReplica.IntTensor(inShape, inInts, 8), packed).logits
    })

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      def tick(): Unit = dut.clockDomain.waitSampling()

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

      dut.io.ctrlBus.aw.valid #= false
      dut.io.ctrlBus.w.valid #= false
      dut.io.ctrlBus.ar.valid #= false
      dut.io.ctrlBus.b.ready #= false
      dut.io.ctrlBus.r.ready #= false
      dut.io.outStream.stream.ready #= true
      tick(); tick()

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
      writeCsr(0x00, 1)

      val outCount = dut.io.outStream.shape.product
      val collected = scala.collection.mutable.ArrayBuffer[Double]()
      var cycles = 0
      val timeout = 20000
      var arBeats = 0L
      var awBeats = 0L
      val ctrlOpt = dut.model.spillCtrl
      val spillIdx = dut.model.spillLayerIdx.get
      val winLoSig = dut.model.spillWinLoOf(spillIdx)
      val aBeatSig = dut.model.spillABeatOf(spillIdx)
      val seedV = dut.model.spillSeedVldOf(spillIdx)
      val seedR = dut.model.spillSeedRdyOf(spillIdx)
      val drainV = dut.model.spillDrainVldOf(spillIdx)
      val drainR = dut.model.spillDrainRdyOf(spillIdx)
      val engDone = dut.model.spillPassDoneOf(spillIdx)
      val aV = dut.model.spillAVldOf(spillIdx)
      val aR = dut.model.spillARdyOf(spillIdx)
      val wV = dut.model.spillWVldOf(spillIdx)
      val wR = dut.model.spillWRdyOf(spillIdx)
      val bV = dut.model.spillBiasVldOf(spillIdx)
      val bR = dut.model.spillBiasRdyOf(spillIdx)
      val yV = dut.model.spillYVldOf(spillIdx)
      val yR = dut.model.spillYRdyOf(spillIdx)
      val cV = dut.model.spillCMonVOf(spillIdx)
      val cR = dut.model.spillCMonROf(spillIdx)
      val eR = dut.model.spillEngReArmOf(spillIdx)
      val bRe = dut.model.spillBiasReArmOf(spillIdx)
      // Cumulative fire counts (levels miss transients at spaced sampling;
      // fires reconstruct exactly how many beats each spill path moved).
      var seedFires = 0L; var drainFires = 0L; var aFires = 0L; var wFires = 0L
      var lastSeedFire = -1; var lastDrainFire = -1; var lastAFire = -1; var lastWFire = -1
      var biasFires = 0L; var yFires = 0L
      var lastBiasFire = -1; var lastYFire = -1
      var cFires = 0L; var lastCFire = -1
      var engReArmHigh = 0L; var biasReArmHigh = 0L
      var engReArmEdges = 0L; var biasReArmEdges = 0L
      var prevER = false; var prevBR = false
      var lastProbe = ""
      var lastArFire = -1
      var lastAwFire = -1
      var lastOutFire = -1

      while (collected.length < outCount && cycles < timeout) {
        // Bus fires: lus chaque cycle (2 lectures combinees, pas de sondes).
        val arFire = dut.io.axiMaster.ar.valid.toBoolean && dut.io.axiMaster.ar.ready.toBoolean
        val awFire = dut.io.axiMaster.aw.valid.toBoolean && dut.io.axiMaster.aw.ready.toBoolean
        if (arFire) {
          arBeats += dut.io.axiMaster.ar.payload.len.toInt + 1
          lastArFire = cycles
        }
        if (awFire) {
          awBeats += dut.io.axiMaster.aw.payload.len.toInt + 1
          lastAwFire = cycles
        }
        val outV = dut.io.outStream.stream.valid.toBoolean
        if (outV) {
          for (l <- 0 until dut.io.outStream.lanes if collected.length < outCount) {
            collected += dut.io.outStream.stream.payload(l).asInstanceOf[SInt].toInt.toDouble
          }
          lastOutFire = cycles
        }
        // Per-cycle spill fire counts (tiny repro: affordable, exact).
        val sV = seedV.toBoolean; val sR = seedR.toBoolean
        val dV = drainV.toBoolean; val dR = drainR.toBoolean
        val av = aV.toBoolean; val ar = aR.toBoolean
        val wv = wV.toBoolean; val wr = wR.toBoolean
        if (sV && sR) { seedFires += 1; lastSeedFire = cycles }
        if (dV && dR) { drainFires += 1; lastDrainFire = cycles }
        if (av && ar) { aFires += 1; lastAFire = cycles }
        if (wv && wr) { wFires += 1; lastWFire = cycles }
        if (bV.toBoolean && bR.toBoolean) { biasFires += 1; lastBiasFire = cycles }
        if (yV.toBoolean && yR.toBoolean) { yFires += 1; lastYFire = cycles }
        if (cV.toBoolean && cR.toBoolean) { cFires += 1; lastCFire = cycles }
        val erNow = eR.toBoolean; val brNow = bRe.toBoolean
        if (erNow) engReArmHigh += 1
        if (brNow) biasReArmHigh += 1
        if (erNow && !prevER) engReArmEdges += 1
        if (brNow && !prevBR) biasReArmEdges += 1
        prevER = erNow; prevBR = brNow
        // Sondes espacees tous les 100 cycles (simPublic controleur + moteur).
        if (cycles % 100 == 0) {
          ctrlOpt.foreach { ctrl =>
            val s = s"pf=${ctrl.io.passFirst.toBoolean} pl=${ctrl.io.passLast.toBoolean} " +
              s"pi=${ctrl.io.passIdx.toInt} rw=${ctrl.io.refetchW.toBoolean} " +
              s"br=${ctrl.io.biasReArm.toBoolean} ra=${ctrl.io.restartA.toBoolean} " +
              s"busy=${ctrl.io.busy.toBoolean} done=${ctrl.io.done.toBoolean} " +
              s"seedV=${seedV.toBoolean} seedR=${seedR.toBoolean} " +
              s"drainV=${drainV.toBoolean} drainR=${drainR.toBoolean} " +
              s"engDone=${engDone.toBoolean} winLo=${winLoSig.toInt} aBeat=${aBeatSig.toInt} " +
              s"aV=${aV.toBoolean} aR=${aR.toBoolean} wV=${wV.toBoolean} wR=${wR.toBoolean} " +
              s"band=${dut.model.imgBandActive.toBoolean}"
            if (s != lastProbe) {
              println(s"SCALE-DBG [$label cyc=$cycles] ctl: $s collected=${collected.length}/$outCount AR=$arBeats AW=$awBeats")
              lastProbe = s
            }
          }
        }
        if (cycles % 2000 == 0 && cycles > 0) {
          println(s"SCALE-DBG [$label cyc=$cycles] progress: collected=${collected.length}/$outCount AR=$arBeats AW=$awBeats " +
            s"lastAr=$lastArFire lastAw=$lastAwFire lastOut=$lastOutFire " +
            s"seedF=$seedFires drainF=$drainFires aF=$aFires wF=$wFires biasF=$biasFires yF=$yFires cF=$cFires " +
            s"engReArmHi=$engReArmHigh/${engReArmEdges}e biasReArmHi=$biasReArmHigh/${biasReArmEdges}e")
        }
        tick(); cycles += 1
      }

      if (collected.length != outCount) {
        // Dump final espace avant echec (diagnostic hang, pas d'attente).
        ctrlOpt.foreach { ctrl =>
          println(s"SCALE-DBG [$label TIMEOUT cyc=$cycles] ctl: pf=${ctrl.io.passFirst.toBoolean} " +
            s"pl=${ctrl.io.passLast.toBoolean} pi=${ctrl.io.passIdx.toInt} rw=${ctrl.io.refetchW.toBoolean} " +
            s"br=${ctrl.io.biasReArm.toBoolean} ra=${ctrl.io.restartA.toBoolean} " +
            s"busy=${ctrl.io.busy.toBoolean} done=${ctrl.io.done.toBoolean} " +
            s"seedV=${seedV.toBoolean} seedR=${seedR.toBoolean} " +
            s"drainV=${drainV.toBoolean} drainR=${drainR.toBoolean} engDone=${engDone.toBoolean} " +
            s"winLo=${winLoSig.toInt} aBeat=${aBeatSig.toInt} " +
            s"aV=${aV.toBoolean} aR=${aR.toBoolean} wV=${wV.toBoolean} wR=${wR.toBoolean} " +
            s"band=${dut.model.imgBandActive.toBoolean}")
        }
        println(s"SCALE-DBG [$label TIMEOUT] bus: ARvalid=${dut.io.axiMaster.ar.valid.toBoolean} " +
          s"ARready=${dut.io.axiMaster.ar.ready.toBoolean} AWvalid=${dut.io.axiMaster.aw.valid.toBoolean} " +
          s"AWready=${dut.io.axiMaster.aw.ready.toBoolean} Wvalid=${dut.io.axiMaster.w.valid.toBoolean} " +
          s"Wready=${dut.io.axiMaster.w.ready.toBoolean} Rvalid=${dut.io.axiMaster.r.valid.toBoolean} " +
          s"Rready=${dut.io.axiMaster.r.ready.toBoolean} Bvalid=${dut.io.axiMaster.b.valid.toBoolean} " +
          s"outValid=${dut.io.outStream.stream.valid.toBoolean}")
        println(s"SCALE-DBG [$label TIMEOUT] collected=${collected.length}/$outCount AR=$arBeats AW=$awBeats " +
          s"weightRegionBytes=$weightBytesTotal lastAr=$lastArFire lastAw=$lastAwFire lastOut=$lastOutFire " +
          s"seedF=$seedFires(last=$lastSeedFire) drainF=$drainFires(last=$lastDrainFire) " +
          s"aF=$aFires(last=$lastAFire) wF=$wFires(last=$lastWFire) " +
          s"biasF=$biasFires(last=$lastBiasFire) yF=$yFires(last=$lastYFire) cF=$cFires(last=$lastCFire) " +
          s"engReArmHi=$engReArmHigh/${engReArmEdges}e biasReArmHi=$biasReArmHigh/${biasReArmEdges}e " +
          s"bV=${bV.toBoolean} bR=${bR.toBoolean} yV=${yV.toBoolean} yR=${yR.toBoolean} " +
          s"cV=${cV.toBoolean} cR=${cR.toBoolean}")
      }
      assert(collected.length == outCount,
        s"[$label] collected ${collected.length}/$outCount outputs in $cycles cycles (AR=$arBeats AW=$awBeats)")

      val dev = collected.toSeq.zip(expected).map { case (h, s) => math.abs(h - s) }.max
      assert(dev == 0.0, s"[$label] replica bit-exact failed: max dev=$dev")
      println(s"SCALE-DBG [$label] PASSED in $cycles cycles (AR=$arBeats AW=$awBeats weightRegionBytes=$weightBytesTotal " +
        s"seedF=$seedFires drainF=$drainFires aF=$aFires wF=$wFires biasF=$biasFires yF=$yFires cF=$cFires " +
        s"engReArmHi=$engReArmHigh/${engReArmEdges}e biasReArmHi=$biasReArmHigh/${biasReArmEdges}e)")
    }
  }

  test("scale-debug control 1-beat (M=1 N=8) passe") {
    runCase(
      Seq(Linear(inFeatures = 8, outFeatures = 8, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), "CTRL-1BEAT")
  }

  test("scale-debug repro 2-beats (M=1 N=16) bloque suspect") {
    runCase(
      Seq(Linear(inFeatures = 16, outFeatures = 16, weightLanes = 2, spillKSlice = 8)),
      Seq(1, 16), "REPRO-2BEATS")
  }
}
