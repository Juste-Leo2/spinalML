package spinalML.heavy

import scala.collection.mutable.ArrayBuffer

/** Deterministic WideResidual parameters (see WideResidual.scala). */
object WideResidualWeights {
  case class Weights(convW3: Seq[Float], convB3: Seq[Float], convW1: Seq[Float], convB1: Seq[Float],
                     fcW: Seq[Seq[Float]], fcB: Seq[Float])

  /** Same generator stream as the original 64x64 constants (side=64 is bit-identical). */
  private def gen(side: Int): Weights = {
    val rng = new scala.util.Random(37217)
    val inF = ((side - 2) / 2) * ((side - 2) / 2)
    Weights(
      Seq.fill(9)(rng.nextFloat() * 2 - 1),
      Seq(rng.nextFloat() * 2 - 1),
      Seq(rng.nextFloat() * 2 - 1),
      Seq(rng.nextFloat() * 2 - 1),
      Seq.fill(10)(Seq.fill(inF)(rng.nextFloat() * 2 - 1)),
      Seq.fill(10)(rng.nextFloat() * 2 - 1))
  }
  def ofSide(side: Int): Weights = gen(side)
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

  def logits(img: Seq[String], w: WideResidualWeights.Weights): Seq[Double] = {
    val side = img.length
    require(img.forall(_.length == side), "wide image must be square side x side")
    val pix = Array.ofDim[Int](side, side)
    for (y <- 0 until side; x <- 0 until side) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

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

    val imgF = Array.tabulate(side, side)((y, x) => bf16Fields(pix(y)(x).toFloat))

    val n1 = convValid(imgF, 3, w.convW3.map(bf16Fields), bf16Fields(w.convB3.head))
    val n2 = relu(n1)
    val n3 = convValid(n2, 1, w.convW1.map(bf16Fields), bf16Fields(w.convB1.head))
    val Hc = side - 2
    val n4 = Array.tabulate(Hc, Hc)((i, j) => fadd(n2(i)(j), n3(i)(j), EB, MB))

    def pool4(nn: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until Hc / 2; j <- 0 until Hc / 2) {
        acts += fmax(fmax(nn(2 * i)(2 * j), nn(2 * i)(2 * j + 1), EB, MB),
          fmax(nn(2 * i + 1)(2 * j), nn(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      }
      acts
    }
    val acts = pool4(n4)

    val out = ArrayBuffer[Double]()
    for (o <- 0 until 10) {
      val row = w.fcW(o).map(bf16Fields)
      val prods = acts.indices.map(k => fmul(acts(k), row(k), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      out += decode(fadd(acc, bf16Fields(w.fcB(o)), EB, MB), EB, MB)
    }
    out.toSeq
  }

  /** Diagnostic variant: pair the tap one element LATER (n2[i+1] + n3[i]). */
  def logitsShifted(img: Seq[String], w: WideResidualWeights.Weights): Seq[Double] = {
    logitsShiftK(img, w, 1)
  }

  /** Diagnostic variant: pair the tap K elements LATER (n2[i+K] + n3[i], wrap). */
  def logitsShiftK(img: Seq[String], w: WideResidualWeights.Weights, shift: Int): Seq[Double] = {
    val side = img.length
    val pix = Array.ofDim[Int](side, side)
    for (y <- 0 until side; x <- 0 until side) pix(y)(x) = if (img(y)(x) == '1') 1 else 0
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
    val imgF = Array.tabulate(side, side)((y, x) => bf16Fields(pix(y)(x).toFloat))
    val n1 = convValid(imgF, 3, w.convW3.map(bf16Fields), bf16Fields(w.convB3.head))
    val n2 = relu(n1)
    val n3 = convValid(n2, 1, w.convW1.map(bf16Fields), bf16Fields(w.convB1.head))
    val Hc = side - 2
    val tot = Hc * Hc
    val n4s = Array.tabulate(Hc, Hc)((i, j) => {
      val flat = i * Hc + j
      val nf = (flat + shift) % tot
      fadd(n2(nf / Hc)(nf % Hc), n3(i)(j), EB, MB)
    })
    def pool4(nn: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until Hc / 2; j <- 0 until Hc / 2)
        acts += fmax(fmax(nn(2 * i)(2 * j), nn(2 * i)(2 * j + 1), EB, MB),
          fmax(nn(2 * i + 1)(2 * j), nn(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      acts
    }
    val acts = pool4(n4s)
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

/** Replica of the tap-free chain (Add replaced by a ReLU) — isolates convK1. */
object WideResidualPlainChainReplica {
  import spinalML.examples.HWFloat._

  val EB = 8
  val MB = 7

  private def bf16Fields(f: Float): F = {
    val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
    F((bits >>> 15 & 1) == 1, (bits >>> 7) & 0xFF, bits & 0x7F)
  }

  def logits(img: Seq[String], w: WideResidualWeights.Weights): Seq[Double] = {
    val side = img.length
    val pix = Array.ofDim[Int](side, side)
    for (y <- 0 until side; x <- 0 until side) pix(y)(x) = if (img(y)(x) == '1') 1 else 0

    val Hc = side - 2
    val np = Hc / 2

    def collect(convOut: Array[Array[F]]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until np; j <- 0 until np) {
        acts += fmax(fmax(convOut(2 * i)(2 * j), convOut(2 * i)(2 * j + 1), EB, MB),
          fmax(convOut(2 * i + 1)(2 * j), convOut(2 * i + 1)(2 * j + 1), EB, MB), EB, MB)
      }
      acts
    }

    val imgF = Array.tabulate(side, side)((y, x) => bf16Fields(pix(y)(x).toFloat))
    val n1 = Array.tabulate(Hc, Hc) { (wy, wx) =>
      val prods = w.convW3.indices.map(k =>
        fmul(imgF(wy + k / 3)(wx + k % 3), bf16Fields(w.convW3(k)), EB, MB))
      val acc = fadd(PZERO, tree(prods, EB, MB), EB, MB)
      fadd(acc, bf16Fields(w.convB3.head), EB, MB)
    }
    val n2 = Array.tabulate(Hc, Hc)((i, j) => if (n1(i)(j).s) PZERO else n1(i)(j))

    val k1 = bf16Fields(w.convW1.head)
    val b1 = bf16Fields(w.convB1.head)
    val n3 = Array.tabulate(Hc, Hc)((i, j) => fadd(fmul(n2(i)(j), k1, EB, MB), b1, EB, MB))
    val n4 = Array.tabulate(Hc, Hc)((i, j) => if (n3(i)(j).s) PZERO else n3(i)(j))

    val acts = collect(n4)
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
