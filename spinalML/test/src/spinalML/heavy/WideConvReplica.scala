package spinalML.heavy

import scala.collection.mutable.ArrayBuffer

/** Deterministic pseudo-random WideConv parameters (see WideConv.scala). */
object WideConvWeights {
  case class Weights(convW: Seq[Float], convB: Seq[Float], fcW: Seq[Seq[Float]], fcB: Seq[Float])

  /** Same generator stream as the original 64x64 constants (side=64 is bit-identical). */
  private def gen(side: Int): Weights = {
    val rng = new scala.util.Random(20176)
    val inF = ((side - 2) / 2) * ((side - 2) / 2)
    Weights(
      Seq.fill(9)(rng.nextFloat() * 2 - 1),
      Seq(rng.nextFloat() * 2 - 1),
      Seq.fill(10)(Seq.fill(inF)(rng.nextFloat() * 2 - 1)),
      Seq.fill(10)(rng.nextFloat() * 2 - 1))
  }
  def ofSide(side: Int): Weights = gen(side)
}

/**
 * Bit-exact software replica of the [[WideConv]] BF16 forward pass
 * (Conv 3x3 -> ReLU -> MaxPool 2x2 -> Flatten -> Linear ->10), same
 * HWFloat conventions as MnistReplica (subnormal-encoded constants follow
 * the hardware zero-class rule, RN-even everywhere).
 */
object WideConvReplica {
  import spinalML.examples.HWFloat._

  val EB = 8 // BF16
  val MB = 7

  private def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  def logits(img: Seq[String], w: WideConvWeights.Weights): Seq[Double] = {
    val side = img.length
    require(img.forall(_.length == side), "wide image must be square side x side")
    val pix = Array.ofDim[Int](side, side)
    for (y <- 0 until side; x <- 0 until side) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    val K = 3
    val Hc = side - K + 1
    val convOut = Array.ofDim[F](Hc, Hc)
    val kernel = w.convW.map(bf16Fields)
    val bias = bf16Fields(w.convB.head)

    for (wy <- 0 until Hc; wx <- 0 until Hc) {
      val prods = for (r <- 0 until K; k <- 0 until K)
        yield fmul(bf16Fields(pix(wy + r)(wx + k).toFloat), kernel(r * K + k), EB, MB)
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      val biased = fadd(acc, bias, EB, MB)
      convOut(wy)(wx) = if (biased.s) PZERO else biased // ReLU
    }

    val np = Hc / 2
    val acts = ArrayBuffer[F]()
    for (i <- 0 until np; j <- 0 until np) {
      val v = fmax(fmax(convOut(2 * i)(2 * j), convOut(2 * i)(2 * j + 1), EB, MB),
        fmax(convOut(2 * i + 1)(2 * j), convOut(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      acts += v
    }
    require(acts.length == np * np)

    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = w.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(w.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }
}
