package spinalML.heavy

import scala.collection.mutable.ArrayBuffer

/** Deterministic pseudo-random WideConv parameters (see WideConv.scala). */
object WideConvWeights {
  private val rng = new scala.util.Random(20176)
  /** Conv 3x3, one output channel, row-major kernel flatten. */
  val convW: Seq[Float] = Seq.fill(9)(rng.nextFloat() * 2 - 1)
  val convB: Seq[Float] = Seq(rng.nextFloat() * 2 - 1)
  /** FC weights: torch-style W^T, one row (961) per output neuron. */
  val fcW: Seq[Seq[Float]] = Seq.fill(10)(Seq.fill(961)(rng.nextFloat() * 2 - 1))
  val fcB: Seq[Float] = Seq.fill(10)(rng.nextFloat() * 2 - 1)
}

/**
 * Bit-exact software replica of the [[WideConv]] BF16 forward pass
 * (Conv 3x3 -> ReLU -> MaxPool 2x2 -> Flatten -> Linear 961->10), same
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

  def logits(img: Seq[String]): Seq[Double] = {
    require(img.length == 64 && img.forall(_.length == 64), "wide image must be 64x64")
    val pix = Array.ofDim[Int](64, 64)
    for (y <- 0 until 64; x <- 0 until 64) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    val K = 3
    val Hc = 62
    val convOut = Array.ofDim[F](Hc, Hc)
    val kernel = WideConvWeights.convW.map(bf16Fields)
    val bias = bf16Fields(WideConvWeights.convB.head)

    for (wy <- 0 until Hc; wx <- 0 until Hc) {
      val prods = for (r <- 0 until K; k <- 0 until K)
        yield fmul(bf16Fields(pix(wy + r)(wx + k).toFloat), kernel(r * K + k), EB, MB)
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      val biased = fadd(acc, bias, EB, MB)
      convOut(wy)(wx) = if (biased.s) PZERO else biased // ReLU
    }

    val acts = ArrayBuffer[F]()
    for (i <- 0 until 31; j <- 0 until 31) {
      val v = fmax(fmax(convOut(2 * i)(2 * j), convOut(2 * i)(2 * j + 1), EB, MB),
        fmax(convOut(2 * i + 1)(2 * j), convOut(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      acts += v
    }
    require(acts.length == 961)

    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = WideConvWeights.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(WideConvWeights.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }
}
