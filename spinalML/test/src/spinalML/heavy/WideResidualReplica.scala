package spinalML.heavy

import scala.collection.mutable.ArrayBuffer

/** Deterministic WideResidual parameters (see WideResidual.scala). */
object WideResidualWeights {
  private val rng = new scala.util.Random(37217)
  /** First conv (3x3, one output channel). */
  val convW3: Seq[Float] = Seq.fill(9)(rng.nextFloat() * 2 - 1)
  val convB3: Seq[Float] = Seq(rng.nextFloat() * 2 - 1)
  /** Skip conv (1x1 identity spatial shape). */
  val convW1: Seq[Float] = Seq(rng.nextFloat() * 2 - 1)
  val convB1: Seq[Float] = Seq(rng.nextFloat() * 2 - 1)
  val fcW: Seq[Seq[Float]] = Seq.fill(10)(Seq.fill(961)(rng.nextFloat() * 2 - 1))
  val fcB: Seq[Float] = Seq.fill(10)(rng.nextFloat() * 2 - 1)
}

/**
 * Bit-exact replica of [[WideResidual]] (Conv3x3 -> ReLU -> Conv1x1 ->
 * Add(skip) -> MaxPool2 -> Flatten -> Linear), HWFloat conventions.
 */
object WideResidualReplica {
  import spinalML.examples.HWFloat._

  val EB = 8
  val MB = 7

  private def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  def logits(img: Seq[String]): Seq[Double] = {
    require(img.length == 64 && img.forall(_.length == 64))
    val pix = Array.ofDim[Int](64, 64)
    for (y <- 0 until 64; x <- 0 until 64) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    def convValid(in: Array[Array[F]], K: Int, kernel: Seq[F], bias: F): Array[Array[F]] = {
      val Hi = in.length
      val Hc = Hi - K + 1
      val out = Array.ofDim[F](Hc, Hc)
      for (wy <- 0 until Hc; wx <- 0 until Hc) {
        val prods = for (r <- 0 until K; k <- 0 until K)
          yield fmul(in(wy + r)(wx + k), kernel(r * K + k), EB, MB)
        val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
        out(wy)(wx) = fadd(acc, bias, EB, MB)
      }
      out
    }

    def relu(in: Array[Array[F]]): Array[Array[F]] =
      Array.tabulate(in.length, in.length)((i, j) => if (in(i)(j).s) PZERO else in(i)(j))

    val imgF = Array.tabulate(64, 64)((y, x) => bf16Fields(pix(y)(x).toFloat))

    val n1 = convValid(imgF, 3, WideResidualWeights.convW3.map(bf16Fields), bf16Fields(WideResidualWeights.convB3.head))
    val n2 = relu(n1)
    val n3 = convValid(n2, 1, WideResidualWeights.convW1.map(bf16Fields), bf16Fields(WideResidualWeights.convB1.head))
    val n4 = Array.tabulate(62, 62)((i, j) => fadd(n2(i)(j), n3(i)(j), EB, MB))
    val n4S = Array.tabulate(62, 62)((i, j) => {
      val nf = (i * 62 + j + 1) % 3844
      fadd(n2(nf / 62)(nf % 62), n3(i)(j), EB, MB)
    })

    def pool4(nn: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until 31; j <- 0 until 31) {
        acts += fmax(fmax(nn(2 * i)(2 * j), nn(2 * i)(2 * j + 1), EB, MB),
          fmax(nn(2 * i + 1)(2 * j), nn(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      }
      acts
    }
    val acts = pool4(n4)
    val actsS = pool4(n4S)

    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = WideResidualWeights.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(WideResidualWeights.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }

  /** Diagnostic variant: pair the tap one element LATER (n2[i+1] + n3[i]). */
  def logitsShifted(img: Seq[String]): Seq[Double] = {
    val pix = Array.ofDim[Int](64, 64)
    for (y <- 0 until 64; x <- 0 until 64) pix(y)(x) = if (img(y)(x) == '1') 1 else 0
    def convValid(in: Array[Array[F]], K: Int, kernel: Seq[F], bias: F): Array[Array[F]] = {
      val Hi = in.length
      val Hc = Hi - K + 1
      val out = Array.ofDim[F](Hc, Hc)
      for (wy <- 0 until Hc; wx <- 0 until Hc) {
        val prods = for (r <- 0 until K; k <- 0 until K)
          yield fmul(in(wy + r)(wx + k), kernel(r * K + k), EB, MB)
        val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
        out(wy)(wx) = fadd(acc, bias, EB, MB)
      }
      out
    }
    def relu(in: Array[Array[F]]): Array[Array[F]] =
      Array.tabulate(in.length, in.length)((i, j) => if (in(i)(j).s) PZERO else in(i)(j))
    val imgF = Array.tabulate(64, 64)((y, x) => bf16Fields(pix(y)(x).toFloat))
    val n1 = convValid(imgF, 3, WideResidualWeights.convW3.map(bf16Fields), bf16Fields(WideResidualWeights.convB3.head))
    val n2 = relu(n1)
    val n3 = convValid(n2, 1, WideResidualWeights.convW1.map(bf16Fields), bf16Fields(WideResidualWeights.convB1.head))
    val n4s = Array.tabulate(62, 62)((i, j) => {
      val flat = i * 62 + j
      val nf = (flat + 1) % 3844
      fadd(n2(nf / 62)(nf % 62), n3(i)(j), EB, MB)
    })
    def pool4(nn: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until 31; j <- 0 until 31)
        acts += fmax(fmax(nn(2 * i)(2 * j), nn(2 * i)(2 * j + 1), EB, MB),
          fmax(nn(2 * i + 1)(2 * j), nn(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      acts
    }
    val acts = pool4(n4s)
    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = WideResidualWeights.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(WideResidualWeights.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }
}

/** Replica of the tap-free chain (Add replaced by a ReLU) — isolates convK1. */
object WideResidualPlainChainReplica {
  import spinalML.examples.HWFloat._

  val EB = 8
  val MB = 7

  private def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  def logits(img: Seq[String]): Seq[Double] = {
    val pix = Array.ofDim[Int](64, 64)
    for (y <- 0 until 64; x <- 0 until 64) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    def collect(convOut: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until 31; j <- 0 until 31) {
        acts += fmax(fmax(convOut(2 * i)(2 * j), convOut(2 * i)(2 * j + 1), EB, MB),
          fmax(convOut(2 * i + 1)(2 * j), convOut(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      }
      acts
    }

    val imgF = Array.tabulate(64, 64)((y, x) => bf16Fields(pix(y)(x).toFloat))
    val n1 = Array.tabulate(62, 62) { (wy, wx) =>
      val prods = WideResidualWeights.convW3.indices.map(k =>
        fmul(bf16Fields(pix(wy + k / 3)(wx + k % 3).toFloat), bf16Fields(WideResidualWeights.convW3(k)), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      fadd(acc, bf16Fields(WideResidualWeights.convB3.head), EB, MB)
    }
    val n2 = Array.tabulate(62, 62)((i, j) => if (n1(i)(j).s) PZERO else n1(i)(j))

    val k1 = bf16Fields(WideResidualWeights.convW1.head)
    val b1 = bf16Fields(WideResidualWeights.convB1.head)
    val n3 = Array.tabulate(62, 62)((i, j) => fadd(fmul(n2(i)(j), k1, EB, MB), b1, EB, MB))
    val n4 = Array.tabulate(62, 62)((i, j) => if (n3(i)(j).s) PZERO else n3(i)(j))

    val acts = collect(n4)
    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = WideResidualWeights.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(WideResidualWeights.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }
}
