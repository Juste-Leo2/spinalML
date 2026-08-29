package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.dtypes.FloatML

/**
 * Minimal, isolated reproduction of the Phase-2b governed-swap corner case:
 * a reduced-lane tile is served while the ADVERSE bank gets a fresh
 * generation in parallel; the end-of-pass flip must never leak stale
 * (or mis-muxed) beats into the served stream. EACH beat of both passes is
 * asserted individually so the exact failing boundary beats are shown.
 */
class SdbSwapDut(depth: Int = 16, lanes: Int = 1) extends Component {
  val io = new Bundle {
    val loadStream   = slave(Stream(Vec(UInt(16 bits), lanes)))
    val outStream    = master(Stream(Vec(UInt(16 bits), lanes)))
    val stageRequest = in Bool() default(False)
    val startGate    = in Bool() default(False)
    val tileFilledOut = out Bool()
    val readProbe    = out UInt(16 bits)
    val rdAddrProbe  = out UInt(10 bits)
    val isRdProbe    = out Bool()
  }

  val sdb = StreamDoubleBuffer(HardType(UInt(16 bits)), depth, lanes, enableFreezePort = true)
  val streamer = DoubleBufferStreamer(HardType(UInt(16 bits)), depth, lanes)

  sdb.io.streamIn << io.loadStream
  sdb.io.nextTile := streamer.io.nextTile
  streamer.io.tileReady := sdb.io.tileReady && io.startGate
  sdb.io.readAddr := streamer.io.readAddr
  streamer.io.readData := sdb.io.readData
  sdb.io.reArm := False
  sdb.io.residentHold.foreach(_ := False)
  sdb.io.stageRequest.foreach(_ := io.stageRequest)
  io.tileFilledOut := sdb.io.tileFilled
  io.readProbe := sdb.io.readData(0)
  io.rdAddrProbe := sdb.io.readAddr
  io.isRdProbe := streamer.io.tileReady
  io.outStream << streamer.io.streamOut
}

class SdbSwapTb extends AnyFunSuite {
  test("governed swap serves full generations (per-beat check)") {
    val beats = 720            // exact W4A8 FC weight: 2880 elements / 4 lanes
    val depth = beats * 4
    val lanes = 4
    val compiled = SimConfig.withVerilator.compile(new SdbSwapDut(depth, lanes))

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      dut.io.loadStream.valid #= false
      dut.io.outStream.ready #= true
      dut.io.stageRequest #= false
      dut.io.startGate #= false

      def push(vals: Seq[Seq[Int]]): Unit = {
        var i = 0
        while (i < vals.size) {
          for (l <- 0 until lanes) dut.io.loadStream.payload(l) #= vals(i)(l)
          dut.io.loadStream.valid #= true
          dut.clockDomain.waitSampling()
          while (!dut.io.loadStream.ready.toBoolean) dut.clockDomain.waitSampling()
          i += 1
        }
        dut.io.loadStream.valid #= false
      }

      // 1) generation 0 fully loaded (bank A)
      push((0 until beats).map(i => (0 until lanes).map(l => i * 4 + l)))
      dut.clockDomain.waitSamplingWhere(dut.io.tileFilledOut.toBoolean)

      // 2) stage the governed swap and fetch generation 1 in parallel (overlap)
      dut.io.stageRequest #= true
      dut.io.startGate #= true
      fork { push((0 until beats).map(i => (0 until lanes).map(l => i * 4 + 30000 + l))) }

      // 3) serve pass 0 — the exact generation-0 window must appear
      val win0 = (0 until beats).map(i => (i * 4 until i * 4 + lanes)).toList
      val stream0 = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
      while (stream0.takeRight(beats).toList != win0) {
        if (stream0.size > 2 * beats + 64) sys.error("pass0 window never appeared")
        if (dut.io.outStream.valid.toBoolean && dut.io.outStream.ready.toBoolean) {
          stream0 += (0 until lanes).map(l => dut.io.outStream.payload(l).toInt)
        }
        dut.clockDomain.waitSampling()
      }
      println(s"pass0 window found")

      // 4) service pass 1: the flip is governed at the end-of-pass nextTile;
      //    the upcoming beats must be generation 1 EXACTLY, incl. the last.
      val stream1 = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
      while (stream1.size < beats) {
        if (dut.io.outStream.valid.toBoolean && dut.io.outStream.ready.toBoolean) {
          stream1 += (0 until lanes).map(l => dut.io.outStream.payload(l).toInt)
        }
        dut.clockDomain.waitSampling()
      }
      val nBad = stream1.zipWithIndex.count { case (v, i) =>
        v != (0 until lanes).map(l => i * 4 + 30000 + l)
      }
      if (nBad == 0) println("pass1 PERFECT")
      assert(nBad == 0, s"pass1 corrupted ($nBad/$beats beats wrong; first bad at ${stream1.zipWithIndex.indexWhere { case (v, i) => v != (0 until lanes).map(l => i * 4 + 30000 + l) }})")
      dut.io.stageRequest #= false
    }
  }
}

// =====================================================================
// Full DDR path microbench: DMAReader (real HW config) -> SDB -> Streamer
// =====================================================================
class DmaSdbDut(axiConfig: Axi4Config) extends Component {
  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest(axiConfig.addressWidth)))
    val axiMaster = master(Axi4(axiConfig))
    val outStream = master(Stream(Vec(FloatML(4, 3), 4)))
    val startGate = in Bool() default(False)
    val tileFilledOut = out Bool()
  }

  val wType = FloatML(4, 3)
  val dma = DMAReader(wType, Seq(10, 288), outLanes = 4, axiConfig,
    trimToElements = true, flushableGearbox = true)
  val sdb = StreamDoubleBuffer(wType, 2880, 4, enableFreezePort = true)
  val streamer = DoubleBufferStreamer(wType, 2880, 4)

  sdb.io.streamIn << dma.io.outStream.stream
  sdb.io.nextTile := streamer.io.nextTile
  streamer.io.tileReady := sdb.io.tileReady && io.startGate
  sdb.io.readAddr := streamer.io.readAddr
  streamer.io.readData := sdb.io.readData
  sdb.io.reArm := io.cmd.fire
  sdb.io.residentHold.foreach(_ := False)
  sdb.io.stageRequest.foreach(_ := False)
  io.tileFilledOut := sdb.io.tileFilled
  io.outStream << streamer.io.streamOut
  io.cmd >> dma.io.cmd
  io.axiMaster.ar << dma.io.axiMaster.ar
  dma.io.axiMaster.r << io.axiMaster.r
  io.axiMaster.aw.valid := False
  io.axiMaster.aw.payload.assignDontCare()
  io.axiMaster.w.valid := False
  io.axiMaster.w.payload.assignDontCare()
  io.axiMaster.b.ready := False
}

class DmaSdbTb extends AnyFunSuite {
  import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}

  def fp8Bits(f: Float): Int = {
    if (f == 0f) return 0
    val sign = if (f < 0) 0x80 else 0
    val a = scala.math.abs(f.toDouble)
    if (a >= scala.math.pow(2, -6)) {
      var e = scala.math.floor(scala.math.log(a) / scala.math.log(2)).toInt
      var mF = a / scala.math.pow(2, e)
      if (mF >= 2) { mF /= 2; e += 1 }
      var m = scala.math.floor((mF - 1) * 8 + 0.5).toInt
      if (m == 8) { m = 0; e += 1 }
      val eb = e + 7
      assert(eb >= 1 && eb <= 15)
      sign | (eb << 3) | m
    } else {
      val m = scala.math.floor(a * 512 + 0.5).toInt
      assert(m <= 7)
      sign | m
    }
  }

  def floatOf(p: Data): Float = {
    val f = p.asInstanceOf[FloatML]
    val eW = f.exponent.getWidth
    val mW = f.mantissa.getWidth
    val sign = if (f.sign.toBoolean) -1 else 1
    val rawE = f.exponent.toInt
    val rawM = f.mantissa.toInt
    val bias = (1 << (eW - 1)) - 1
    val mag =
      if (rawE == 0) rawM * scala.math.pow(2, 1 - bias - mW)
      else (1 + rawM.toDouble / (1 << mW)) * scala.math.pow(2, rawE - bias)
    (sign * mag).toFloat
  }

  // FLAKY — seed-dependent race in DmaSdbDut (streamer repeats beat 0:
  // 2832/2880 with seed 2094212935/2014737886, exact with 1481499482).
  // See docs/bugs/2026-08-prefetch-eager-stale-fifo-session.md §5bis.
  // TODO: de-flake after the LUT re-implementation (probe cmd.fire /
  // reArm / startGate / readAddr in the DUT).
  ignore("DMA->SDB full path serves the W4A8 FC weight exactly") {
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    val fw = Mnistw4a8Weights.fcW.flatten // 2880 floats
    val bytes = fw.map(fp8Bits)
    val words = bytes.grouped(8).map(g => (g.zipWithIndex.foldLeft(BigInt(0))((acc, e) =>
      acc | (BigInt(e._1 & 0xFF) << (8 * e._2)))).toBigInt).toSeq
    assert(words.size == 360)
    val base = 0x20000L

    val compiled = SimConfig.withVerilator.compile(new DmaSdbDut(axiConfig))
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      dut.io.startGate #= false
      dut.io.cmd.valid #= false
      val memorySim = AxiMemorySim(axi = dut.io.axiMaster, clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      for ((w, i) <- words.zipWithIndex) memorySim.memory.writeBigInt(base + i * 8, w, 8)

      dut.io.cmd.address #= base
      dut.io.cmd.length #= 359
      dut.io.cmd.valid #= true
      dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
      dut.io.cmd.valid #= false

      dut.clockDomain.waitSamplingWhere(dut.io.tileFilledOut.toBoolean)
      dut.io.startGate #= true
      val got = scala.collection.mutable.ArrayBuffer[Float]()
      while (got.size < 2880) {
        if (dut.io.outStream.valid.toBoolean) {
          for (l <- 0 until 4) got += floatOf(dut.io.outStream.payload(l))
        }
        dut.clockDomain.waitSampling()
      }
      val nBad = got.zipWithIndex.count { case (v, i) => scala.math.abs(v - fw(i)) > 1e-4 }
      println(s"DMA-path decode result: $nBad/2880 elements differ")
      assert(nBad == 0, s"$nBad elements wrong; first bad at ${got.zipWithIndex.indexWhere { case (v, i) => scala.math.abs(v - fw(i)) > 1e-4 }}")
    }
  }
}
