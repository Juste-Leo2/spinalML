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

  /** Diagnostic variant (fork-replay hypothesis): the conv-K1 branch got the
   *  first node-2 element twice (backpressure hold between the first beat and
   *  the rest), so its output b is one element late: n4[k] = n2[k] + n3[k-1]
   *  (k=0 stays n2[0]+n3[0]). */
  def logitsShiftB(img: Seq[String], w: WideResidualWeights.Weights): Seq[Double] = {
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
    val n2f = n2.flatten
    val n3f = n3.flatten
    val n4s: Array[F] = Array.ofDim[F](n2f.length)
    n4s(0) = fadd(n2f(0), n3f(0), EB, MB)
    for (i <- 1 until n2f.length) n4s(i) = fadd(n2f(i), n3f(i - 1), EB, MB)
    val np = Hc / 2
    def pool4(nn: Array[F]): ArrayBuffer[F] = {
      val acts = ArrayBuffer[F]()
      for (i <- 0 until np; j <- 0 until np)
        acts += fmax(fmax(nn((2 * i) * Hc + 2 * j), nn((2 * i) * Hc + 2 * j + 1), EB, MB),
          fmax(nn((2 * i + 1) * Hc + 2 * j), nn((2 * i + 1) * Hc + 2 * j + 1), EB, MB), EB, MB)
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


  /** Diagnostic variant (replay-seam hypothesis): the conv-K1 direct input got
   *  n2[s] twice (its trailing VALID beat re-served at the first full-stall
   *  deep inside the stream), so its entire output is one element late from
   *  k = s+1 on: n4[k] = n2[k] + n3[k-1] for k > s, n4[k] = n2[k] + n3[k] else. */
  def logitsShiftSeam(img: Seq[String], w: WideResidualWeights.Weights, seam: Int): Seq[Double] = {
    val side = img.length
    val pix = Array.ofDim[Float](side * side)
    for (y <- 0 until side; x <- 0 until side) pix(y * side + x) = (if (img(y)(x) == '1') 1 else 0).toFloat

    def convValid(in: IndexedSeq[F], K: Int, inSide: Int, kernel: Seq[F], bias: F): IndexedSeq[F] = {
      val Hc = inSide - K + 1
      val out = Array.ofDim[F](Hc * Hc)
      var wy = 0
      while (wy < Hc) {
        var wx = 0
        while (wx < Hc) {
          val terms = for (r <- 0 until K; k <- 0 until K)
            yield fmul(in((wy + r) * inSide + (wx + k)), kernel(r * K + k), EB, MB)
          out(wy * Hc + wx) = fadd(fadd(PZERO, tree(terms, EB, MB), EB, MB), bias, EB, MB)
          wx += 1
        }
        wy += 1
      }
      out.toIndexedSeq
    }

    val imgF = pix.map(x => bf16Fields(x))
    val n1 = convValid(imgF, 3, side, w.convW3.map(bf16Fields), bf16Fields(w.convB3.head))
    val n2 = n1.map(f => if (f.s) PZERO else f)
    val n3 = convValid(n2, 1, side - 2, w.convW1.map(bf16Fields), bf16Fields(w.convB1.head))
    val Hc = side - 2
    val total = Hc * Hc
    val n4 = Array.ofDim[F](total)
    for (k <- 0 until total) {
      if (k > seam && k - 1 < total) n4(k) = fadd(n2(k), n3(k - 1), EB, MB)
      else n4(k) = fadd(n2(k), n3(k), EB, MB)
    }
    val np = Hc / 2
    def pool4(nn: IndexedSeq[F]): scala.collection.mutable.ArrayBuffer[F] = {
      val acts = scala.collection.mutable.ArrayBuffer[F]()
      for (i <- 0 until np; j <- 0 until np)
        acts += fmax(fmax(nn((2 * i) * Hc + 2 * j), nn((2 * i) * Hc + 2 * j + 1), EB, MB),
          fmax(nn((2 * i + 1) * Hc + 2 * j), nn((2 * i + 1) * Hc + 2 * j + 1), EB, MB), EB, MB)
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

  /** Exhaustive seam hunt: returns (seam, maxDev) sorted by dev for all seams. */
  def logitsShiftSeamBest(img: Seq[String], w: WideResidualWeights.Weights, hw: Seq[Double]): Seq[(Int, Double)] = {
    val total = (img.length - 2) * (img.length - 2)
    (0 until total).map { s =>
      val ref = logitsShiftSeam(img, w, s)
      val dev = hw.zip(ref).map { case (h, r) => java.lang.Math.abs(h - r) }.max
      (s, dev)
    }.sortBy(_._2)
  }

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
