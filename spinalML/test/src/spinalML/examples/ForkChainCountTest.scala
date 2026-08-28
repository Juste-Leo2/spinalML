package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.BF16
import spinalML.dtypes.FloatML
import spinalML.layers.Conv2D
import spinalML.activations.relu
import spinalML.ops.add
import spinalML.poolings.maxpool2d
import spinalML.memory.TapBuffer
import scala.collection.mutable.ArrayBuffer
import spinalML.utils.SimLog
import spinalML.examples.{HWFloat, MnistReplica}
import HWFloat.{F, PZERO, decode, fmul, fadd, tree}
import MnistReplica.bf16Fields

/** Exact WIDE-residual spine at the component level, with in-between counters:
 *  image -> Conv3x3 -> ReLU -> TapBuffer.fork(2) -> [Conv1x1 -> repA] + [tap]
 *  -> Add -> MaxPool2x2 -> drain. Records every stream's payload at each fire
 *  so the exact beat/ordering pattern can be diffed against a Scala reference
 *  emulated in BF16 (HWFloat = bit-exact, validated by the PLAIN network). */
case class ForkChainCountComp(H: Int, W: Int, K: Int) extends Component {
  val io = new Bundle {
    val x     = slave(Tensor(BF16(), Seq(H, W), lanes = 1))
    val w3    = slave(Tensor(BF16(), Seq(K * K, 1), lanes = K * K))
    val b3    = slave(Tensor(BF16(), Seq(1, 1), lanes = 1))
    val w1    = slave(Tensor(BF16(), Seq(1, 1), lanes = 1))
    val b1    = slave(Tensor(BF16(), Seq(1, 1), lanes = 1))
    val done  = out Bool()
  }

  val n1 = Conv2D(io.x, io.w3, io.b3, BF16(), false)
  val n2 = relu(n1)

  val Hc = H - K + 1
  SimLog.debug("FORK")(s"n2 shape=${n2.shape} lanes=${n2.lanes} elements=${n2.shape.product}")
  val tee = TapBuffer(BF16(), 196, 1, spineDebug = true)
  tee.io.streamIn << n2.stream
  val tiles = Seq(
    Tensor(BF16(), n2.shape, n2.lanes),
    Tensor(BF16(), n2.shape, n2.lanes))
  tiles(0).stream << tee.io.directOut
  tiles(1).stream << tee.io.tapOut
  val n3 = Conv2D(tiles(0), io.w1, io.b1, BF16(), false)
  val n4 = add(tiles(1), n3)
  val pooled = maxpool2d(n4, 2, 2)
  tee.io.dbg.simPublic()

  val drain = Stream(Vec(BF16(), pooled.lanes))
  drain.ready := True
  drain << pooled.stream

  io.done := drain.valid && drain.ready

  val publicStreams = Seq(io.x.stream, n1.stream, n2.stream,
    tiles(0).stream, tiles(1).stream, n3.stream, n4.stream)
  publicStreams.foreach(s => { s.simPublic(); s.payload.head.simPublic() })
}

class ForkChainCountTest extends AnyFunSuite {
  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384,
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = BOOT))

  private def setF(p: FloatML, bits: Int): Unit = {
    p.sign #= (bits >> 15 & 1) == 1
    p.exponent #= (bits >> 7) & 0xFF
    p.mantissa #= bits & 0x7F
  }
  private def getF(p: FloatML): Int = {
    ((if (p.sign.toBoolean) 1 else 0) << 15) | ((p.exponent.toInt & 0xFF) << 7) | (p.mantissa.toInt & 0x7F)
  }
  val EB = 8
  val MB = 7
  private def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
  private def decoded(bits: Int): Double = decode(F((bits & 0x8000) != 0, (bits >> 7) & 0xFF, bits & 0x7F), EB, MB)

  private def sendOne(dut: ForkChainCountComp, stream: Stream[Vec[FloatML]], bits: Int): Unit = {
    var done = false
    while (!done) {
      stream.valid #= true
      for (i <- stream.payload.indices) setF(stream.payload(i), bits)
      dut.clockDomain.waitSampling()
      if (stream.ready.toBoolean) done = true
    }
    stream.valid #= false
  }

  /** reference: conv3 -> relu -> conv1 on the given pixels (HWFloat emulation) */
  private def refChain(x: IndexedSeq[Float], H: Int, W: Int, K3: Int): (IndexedSeq[F], IndexedSeq[F]) = {
    val Hc = H - K3 + 1
    val w3: Seq[F] = (0 until K3 * K3).map(_ => bf16Fields(0.5f))
    val b3: F      = bf16Fields(0.0f)
    val w1: Seq[F] = Seq(bf16Fields(0.5f))
    val b1: F      = bf16Fields(0.0f)

    val n1 = Array.ofDim[F](Hc * Hc)
    for (k <- 0 until Hc * Hc) {
      val wy = k / Hc
      val wx = k % Hc
      val terms = for (r <- 0 until K3; c <- 0 until K3) yield fmul(bf16Fields(x((wy + r) * W + (wx + c))), w3(r * K3 + c), EB, MB)
      n1(k) = fadd(tree(terms.toSeq, EB, MB), b3, EB, MB)
    }
    val n2 = Array.ofDim[F](Hc * Hc)
    for (k <- 0 until Hc * Hc) n2(k) = if (n1(k).s) PZERO else n1(k)
    val n3 = Array.ofDim[F](Hc * Hc)
    for (k <- 0 until Hc * Hc) n3(k) = fadd(fmul(n2(k), w1.head, EB, MB), b1, EB, MB)
    (n2.toIndexedSeq, n3.toIndexedSeq)
  }

  test("ForkChain: per-node payload sequence on one 16x16 frame") {
    SimLog.bench("ForkChainCountTest", "FORK") {
      val H = 16
    val W = 16
    val K = 3
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
      .compile(ForkChainCountComp(H, W, K))
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      for (s <- Seq(dut.io.x, dut.io.w3, dut.io.b3, dut.io.w1, dut.io.b1)) {
        s.stream.valid #= false
      }
      dut.clockDomain.waitSampling(3)

      sendOne(dut, dut.io.w3.stream, bf16Bits(0.5f))
      sendOne(dut, dut.io.b3.stream, bf16Bits(0.0f))
      sendOne(dut, dut.io.w1.stream, bf16Bits(0.5f))
      sendOne(dut, dut.io.b1.stream, bf16Bits(0.0f))

      val streams = Seq[(String, Stream[Vec[FloatML]])](
        "x"      -> dut.io.x.stream,
        "n1"     -> dut.n1.stream,
        "n2"     -> dut.n2.stream,
        "direct" -> dut.tiles(0).stream,
        "tap"    -> dut.tiles(1).stream,
        "n3"     -> dut.n3.stream,
        "n4"     -> dut.n4.stream
      )
      val recs: Array[ArrayBuffer[Int]] = Array.fill(streams.size)(ArrayBuffer[Int]())
      val recAt: Array[ArrayBuffer[Long]] = Array.fill(streams.size)(ArrayBuffer[Long]())

      var xIdx = 0
      val total = H * W
      val pushRec = ArrayBuffer[Int]()
      val popRec = ArrayBuffer[Int]()
      val pushTime = ArrayBuffer[Long]()
      val popTime = ArrayBuffer[Long]()
      var doneCount = 0
      var cycles = 0L
      while (doneCount < 49 && cycles < 50000) {
        var i = 0
        while (i < streams.size) {
          val s = streams(i)._2
          if (s.valid.toBoolean && s.ready.toBoolean) { recs(i) += getF(s.payload(0)); recAt(i) += cycles }
          i += 1
        }
        if (xIdx < total) {
          dut.io.x.stream.valid #= true
          setF(dut.io.x.stream.payload(0), bf16Bits((xIdx + 1).toFloat))
          dut.clockDomain.waitSampling()
          if (dut.io.x.stream.ready.toBoolean) xIdx += 1
        } else {
          dut.io.x.stream.valid #= false
          dut.clockDomain.waitSampling()
        }
          if (dut.tee.io.dbg.pushFire.toBoolean) {
          pushRec += dut.tee.io.dbg.pushVal.toInt
          pushTime += cycles
        }
        if (dut.tee.io.dbg.popFire.toBoolean) {
          popRec += dut.tee.io.dbg.popVal.toInt
          popTime += cycles
        }
        if (dut.io.done.toBoolean) doneCount += 1
        cycles += 1
        if (cycles % 10000 == 0) {
          SimLog.debug("FORK")(s"[t=$cycles] x=$xIdx done=$doneCount lens=" +
            recs.map(_.size).mkString(","))
        }
      }
      dut.io.x.stream.valid #= false
      dut.clockDomain.waitSampling(20)
      SimLog.debug("FORK")(s"[t=$cycles] drained done=$doneCount lens=" + recs.map(_.size).mkString(","))

      val names = streams.map(_._1)
      val (n2ref, n3ref) = refChain((1 to total).map(_.toFloat), H, W, K)

      SimLog.info("FORK")("== lengths ==")
      for (i <- names.indices) SimLog.info("FORK")(f"  ${names(i)}%-7s len=${recs(i).size}")

      def idxOf(seq: Seq[F], v: Int): Int =
        seq.zipWithIndex.minBy { case (w, _) => java.lang.Math.abs(decode(w, EB, MB) - decoded(v)) }._2

      val direct = recs(names.indexOf("direct"))
      SimLog.info("FORK")(s"  PUSH seq [first 14] = " + pushRec.take(14).map(decoded _).mkString(", "))
      SimLog.info("FORK")(s"  POP  seq [first 14] = " + popRec.take(14).map(decoded _).mkString(", "))
      SimLog.info("FORK")(s"  PUSH times first   = " + pushTime.take(14).mkString(","))
      SimLog.info("FORK")(s"  POP  times first   = " + popTime.take(14).mkString(","))
      SimLog.info("FORK")(s"  len(pop)=${popRec.size} len(push)=${pushRec.size} tail pop=" + popRec.takeRight(8).map(decoded _).mkString(", "))
      val n2seq = recs(names.indexOf("n2"))
      val n1seq = recs(names.indexOf("n1"))
      SimLog.info("FORK")("== source raw records ==")
      SimLog.info("FORK")(s"  n1[0..9] = ${n1seq.take(10).map(decoded _).mkString(", ")}")
      SimLog.info("FORK")(s"  n2[0..9] = ${n2seq.take(10).map(decoded _).mkString(", ")}")
      SimLog.info("FORK")(s"  n2 first/cycles = ${n2seq.zip(recAt(names.indexOf("n2"))).take(6).map { case (v, t) => f"$v%.0f@$t" }.mkString(", ")}")
      SimLog.info("FORK")(s"  direct first/cycles = ${direct.zip(recAt(names.indexOf("direct"))).take(6).map { case (v, t) => f"$v%.0f@$t" }.mkString(", ")}")
      SimLog.info("FORK")(s"  direct last/cycles = ${direct.zip(recAt(names.indexOf("direct"))).takeRight(6).map { case (v, t) => f"$v%.0f@$t" }.mkString(", ")}")
      SimLog.info("FORK")("== direct (fork -> conv1) vs n2ref ==")
      SimLog.info("FORK")(s"  direct[0..7]      = ${direct.take(8).map(decoded).mkString(", ")}")
      SimLog.info("FORK")(s"  n2ref [0..7]      = ${n2ref.take(8).map(x => decode(x, EB, MB)).mkString(", ")}")
      val map = direct.map(v => idxOf(n2ref, v))
      SimLog.info("FORK")(s"  direct->n2idx[0..9] = ${map.take(10).mkString(", ")}")
      val firstDiff = map.zipWithIndex.find { case (idx, j) => idx != j }
      SimLog.info("FORK")(s"  first diff at j=$firstDiff (expected idx==j)")
      SimLog.info("FORK")(s"  direct->n2idx tail  = ${map.takeRight(10).mkString(", ")}")

      val n3seq = recs(names.indexOf("n3"))
      val n3map = n3seq.map(v => idxOf(n3ref, v))
      SimLog.info("FORK")(s"  n3->idx[0..9]    = ${n3map.take(10).mkString(", ")}  tail=${n3map.takeRight(10).mkString(", ")}")

      val tap = recs(names.indexOf("tap"))
      val n4seq = recs(names.indexOf("n4"))
      var mism = 0
      val maxK = java.lang.Math.min(n4seq.size, java.lang.Math.min(tap.size, n3seq.size))
      for (k <- 0 until maxK) {
        val refPair = fadd(bf16Fields(decoded(tap(k)).toFloat), bf16Fields(decoded(n3seq(k)).toFloat), EB, MB)
        if (refPair != bf16Fields(decoded(n4seq(k)).toFloat)) mism += 1
      }
      SimLog.info("FORK")("== pairs ==")
      SimLog.info("FORK")(s"  n4 = tap + n3 mismatch count = $mism / $maxK")
      for (k <- 0 until java.lang.Math.min(6, maxK))
        SimLog.info("FORK")(f"  pair$k: tap=${decoded(tap(k))}%.6f n3=${decoded(n3seq(k))}%.6f n4=${decoded(n4seq(k))}%.6f")
    }
    }
  }
}
