package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.nn._
import HWArithmetic._

/**
 * Universal interpreter of Seq[LayerSpec] computing the bit-exact software oracle
 * and printing intermediate activation previews (first 3 and last 3 elements, min/max).
 */
object ModelReplica {

  sealed trait ReplicaTensor {
    def shape: Seq[Int]
    def toDoubles: Seq[Double]
    def asInts: Seq[Long]
    def asFloats: Seq[F]
    def length: Int
  }

  case class IntTensor(shape: Seq[Int], data: Seq[Long], bitWidth: Int) extends ReplicaTensor {
    def toDoubles: Seq[Double] = data.map(_.toDouble)
    def asInts: Seq[Long] = data
    def asFloats: Seq[F] = throw new UnsupportedOperationException("IntTensor is not a float tensor")
    def length: Int = data.length
  }

  case class FloatTensor(shape: Seq[Int], data: Seq[F], expBits: Int, mantBits: Int) extends ReplicaTensor {
    def toDoubles: Seq[Double] = data.map(f => decode(f, expBits, mantBits))
    def asInts: Seq[Long] = throw new UnsupportedOperationException("FloatTensor is not an int tensor")
    def asFloats: Seq[F] = data
    def length: Int = data.length
  }

  case class LayerExecutionTrace(
    layerIdx: Int,
    layerName: String,
    outShape: Seq[Int],
    first3: Seq[Double],
    last3: Seq[Double],
    minVal: Double,
    maxVal: Double,
    tensor: Seq[F] = Nil,
    replicaTensor: Option[ReplicaTensor] = None
  )

  case class ForwardResult(
    logits: Seq[Double],
    traces: Seq[LayerExecutionTrace]
  )

  private def formatPreview(s: Seq[Double]): String = {
    s.map(v => f"$v%6.3f").mkString("[", ", ", "]")
  }

  /**
   * Universal forward pass execution on any Accelerator / Sequential model using ReplicaTensor.
   */
  def forwardWithTrace(
    layers: Seq[LayerSpec],
    inputShape: Seq[Int],
    inputTensor: ReplicaTensor,
    packedWeights: WeightMemoryLayout.PackedWeightsResult
  ): ForwardResult = {

    println("\n" + "=" * 80)
    println("Model Architecture & Intermediate Activation Trace:")
    println("-" * 80)

    val inDoubles = inputTensor.toDoubles
    val inFirst3 = inDoubles.take(3)
    val inLast3 = inDoubles.takeRight(3)
    val inMin = if (inDoubles.nonEmpty) inDoubles.min else 0.0
    val inMax = if (inDoubles.nonEmpty) inDoubles.max else 0.0

    println(f"Input Tensor        : shape ${inputShape.mkString("x")}%-12s | first3=${formatPreview(inFirst3)} ... last3=${formatPreview(inLast3)} | min=$inMin%6.3f, max=$inMax%6.3f")

    // Node storage for DAG execution
    val nodeOutputs = ArrayBuffer[ReplicaTensor]()
    val traces = ArrayBuffer[LayerExecutionTrace]()

    nodeOutputs += inputTensor

    var curShape = inputShape
    var curTensor = inputTensor

    for (i <- layers.indices) {
      val layer = layers(i)
      val wInfo = packedWeights.layers(i)
      val lName = layer.getClass.getSimpleName

      var nextShape = curShape
      var nextTensor: ReplicaTensor = curTensor

      layer match {
        case c: Conv2D =>
          require(curShape.length >= 2, "Conv2D requires 2D or 3D input shape")
          val h = curShape(0)
          val w = curShape(1)
          val inC = if (curShape.length >= 3) curShape(2) else 1
          val kSize = c.kernelSize
          val outC = c.outChannels
          val kElems = kSize * kSize * inC
          val hOut = h - kSize + 1
          val wOut = w - kSize + 1
          nextShape = Seq(hOut, wOut, outC)

          curTensor match {
            case it: IntTensor =>
              val arr3D = Array.ofDim[Long](inC, h, w)
              var idx = 0
              val raw = it.asInts
              for (y <- 0 until h; x <- 0 until w; ch <- 0 until inC) {
                arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else 0L
                idx += 1
              }
              val convW = (0 until outC).map(o => wInfo.weightInts.slice(o * kElems, (o + 1) * kElems))
              val convB = (0 until outC).map(o => if (o < wInfo.biasInts.length) wInfo.biasInts(o) else 0L)
              val convOut = LayerReplicas.conv2DInt(arr3D, convW, convB, inC, outC, kSize)
              val flat = LayerReplicas.flattenInt(convOut)
              val outWidth = if (wInfo.biasDtype != null) wInfo.biasDtype().getBitsWidth else 16
              nextTensor = IntTensor(nextShape, flat, outWidth)

            case ft: FloatTensor =>
              val arr3D = Array.ofDim[F](inC, h, w)
              var idx = 0
              val raw = ft.asFloats
              for (y <- 0 until h; x <- 0 until w; ch <- 0 until inC) {
                arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else PZERO
                idx += 1
              }
              val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
              val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
              val convOut = LayerReplicas.conv2D(arr3D, convW, convB, inC, outC, kSize, ft.expBits, ft.mantBits)
              val flat = LayerReplicas.flatten(convOut)
              nextTensor = FloatTensor(nextShape, flat, ft.expBits, ft.mantBits)
          }

        case c: Conv1D =>
          val l = curShape(0)
          val inC = if (curShape.length >= 2) curShape(1) else 1
          val kSize = c.kernelSize
          val outC = c.outChannels
          val kElems = kSize * inC
          nextShape = Seq(l - kSize + 1, outC)

          curTensor match {
            case ft: FloatTensor =>
              val arr2D = Array.ofDim[F](l, inC)
              var idx = 0
              val raw = ft.asFloats
              for (pos <- 0 until l; ch <- 0 until inC) {
                arr2D(pos)(ch) = if (idx < raw.length) raw(idx) else PZERO
                idx += 1
              }
              val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
              val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
              val convOut = LayerReplicas.conv1D(arr2D, convW, convB, inC, outC, kSize, ft.expBits, ft.mantBits)
              val flat = ArrayBuffer[F]()
              for (pos <- convOut.indices; ch <- 0 until outC) flat += convOut(pos)(ch)
              nextTensor = FloatTensor(nextShape, flat.toSeq, ft.expBits, ft.mantBits)
            case it: IntTensor =>
              throw new UnsupportedOperationException("Conv1D in int domain not supported")
          }

        case _: ReLU =>
          curTensor match {
            case it: IntTensor =>
              nextTensor = IntTensor(curShape, LayerReplicas.reluInt(it.asInts), it.bitWidth)
            case ft: FloatTensor =>
              nextTensor = FloatTensor(curShape, LayerReplicas.relu1D(ft.asFloats), ft.expBits, ft.mantBits)
          }

        case lr: LeakyReLU =>
          curTensor match {
            case ft: FloatTensor =>
              nextTensor = FloatTensor(curShape, LayerReplicas.leakyRelu(ft.asFloats, lr.shift, ft.expBits, ft.mantBits), ft.expBits, ft.mantBits)
            case it: IntTensor =>
              throw new UnsupportedOperationException("LeakyReLU in int domain not supported")
          }

        case p: MaxPool2D =>
          val h = curShape(0); val w = curShape(1); val c = if (curShape.length >= 3) curShape(2) else 1
          val hOut = (h - p.poolSize) / p.stride + 1
          val wOut = (w - p.poolSize) / p.stride + 1
          nextShape = Seq(hOut, wOut, c)

          curTensor match {
            case it: IntTensor =>
              val arr3D = Array.ofDim[Long](c, h, w)
              var idx = 0
              val raw = it.asInts
              for (y <- 0 until h; x <- 0 until w; ch <- 0 until c) {
                arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else 0L
                idx += 1
              }
              val pooled = LayerReplicas.maxPool2DInt(arr3D, p.poolSize, p.stride)
              val flat = LayerReplicas.flattenInt(pooled)
              nextTensor = IntTensor(nextShape, flat, it.bitWidth)

            case ft: FloatTensor =>
              val arr3D = Array.ofDim[F](c, h, w)
              var idx = 0
              val raw = ft.asFloats
              for (y <- 0 until h; x <- 0 until w; ch <- 0 until c) {
                arr3D(ch)(y)(x) = if (idx < raw.length) raw(idx) else PZERO
                idx += 1
              }
              val pooled = LayerReplicas.maxPool2D(arr3D, p.poolSize, p.stride, ft.expBits, ft.mantBits)
              val flat = LayerReplicas.flatten(pooled)
              nextTensor = FloatTensor(nextShape, flat, ft.expBits, ft.mantBits)
          }

        case p: MaxPool1D =>
          val l = curShape(0); val c = if (curShape.length >= 2) curShape(1) else 1
          val lOut = (l - p.poolSize) / p.stride + 1
          nextShape = Seq(lOut, c)
          val ft = curTensor.asInstanceOf[FloatTensor]
          val arr2D = Array.ofDim[F](l, c)
          var idx = 0
          for (pos <- 0 until l; ch <- 0 until c) {
            arr2D(pos)(ch) = if (idx < ft.asFloats.length) ft.asFloats(idx) else PZERO
            idx += 1
          }
          val pooled = LayerReplicas.maxPool1D(arr2D, p.poolSize, p.stride, ft.expBits, ft.mantBits)
          val flat = ArrayBuffer[F]()
          for (pos <- pooled.indices; ch <- 0 until c) flat += pooled(pos)(ch)
          nextTensor = FloatTensor(nextShape, flat.toSeq, ft.expBits, ft.mantBits)

        case c: Cast =>
          val targetData = c.targetType()
          curTensor match {
            case it: IntTensor =>
              if (targetData.isInstanceOf[spinalML.dtypes.FloatML]) {
                val fType = targetData.asInstanceOf[spinalML.dtypes.FloatML]
                val tExp = fType.expBits
                val tMant = fType.mantBits
                val converted = LayerReplicas.castIntToFloat(it.asInts, it.bitWidth, tExp, tMant, c.scales)
                nextTensor = FloatTensor(curShape, converted, tExp, tMant)
              } else {
                nextTensor = IntTensor(curShape, it.asInts, targetData.getBitsWidth)
              }
            case ft: FloatTensor =>
              if (targetData.isInstanceOf[spinalML.dtypes.FloatML]) {
                val fType = targetData.asInstanceOf[spinalML.dtypes.FloatML]
                val tExp = fType.expBits
                val tMant = fType.mantBits
                val scale = if (c.scales.nonEmpty) c.scales.head else 1.0
                val converted = LayerReplicas.cast(ft.asFloats, scale, ft.expBits, ft.mantBits, tExp, tMant)
                nextTensor = FloatTensor(curShape, converted, tExp, tMant)
              } else {
                throw new UnsupportedOperationException("Cast from Float to Int not supported in replica")
              }
          }

        case _: Flatten =>
          nextShape = Seq(1, curShape.product)
          curTensor match {
            case it: IntTensor => nextTensor = IntTensor(nextShape, it.asInts, it.bitWidth)
            case ft: FloatTensor => nextTensor = FloatTensor(nextShape, ft.asFloats, ft.expBits, ft.mantBits)
          }

        case l: Linear =>
          val inFeatures = l.inFeatures
          val outFeatures = l.outFeatures
          val lanes = l.effLanes
          nextShape = Seq(1, outFeatures)
          curTensor match {
            case ft: FloatTensor =>
              val fcW = (0 until outFeatures).map(o => wInfo.weightValues.slice(o * inFeatures, (o + 1) * inFeatures))
              val fcB = (0 until outFeatures).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)
              val out = LayerReplicas.linear(ft.asFloats, fcW, fcB, ft.expBits, ft.mantBits, lanes)
              nextTensor = FloatTensor(nextShape, out, ft.expBits, ft.mantBits)
            case it: IntTensor =>
              val raw = it.asInts
              val outBits = if (wInfo.biasDtype != null) wInfo.biasDtype().getBitsWidth else it.bitWidth
              def wrap(v: Long): Long = {
                if (outBits >= 64) v
                else {
                  val mask = (1L << outBits) - 1
                  val unsigned = v & mask
                  if ((unsigned & (1L << (outBits - 1))) != 0) unsigned - (1L << outBits) else unsigned
                }
              }
              val outInts = (0 until outFeatures).map { o =>
                val fcW = wInfo.weightInts.slice(o * inFeatures, (o + 1) * inFeatures)
                val fcB = if (o < wInfo.biasInts.length) wInfo.biasInts(o) else 0L
                var acc = fcB
                for (k <- 0 until inFeatures) {
                  val v = if (k < raw.length) raw(k) else 0L
                  val w = if (k < fcW.length) fcW(k) else 0L
                  acc += v * w
                }
                wrap(acc)
              }
              nextTensor = IntTensor(nextShape, outInts, outBits)
          }

        case bn: BatchNorm1D =>
          val feat = bn.features
          val ft = curTensor.asInstanceOf[FloatTensor]
          val gamma = wInfo.weightValues.take(feat)
          val beta = wInfo.biasValues.take(feat)
          val out = LayerReplicas.batchNorm1D(ft.asFloats, gamma, beta, ft.expBits, ft.mantBits)
          nextTensor = FloatTensor(curShape, out, ft.expBits, ft.mantBits)

        case ad: Add =>
          val ta = nodeOutputs(ad.a)
          val tb = nodeOutputs(ad.b)
          (ta, tb) match {
            case (fa: FloatTensor, fb: FloatTensor) =>
              val out = LayerReplicas.add(fa.asFloats, fb.asFloats, fa.expBits, fa.mantBits)
              nextTensor = FloatTensor(curShape, out, fa.expBits, fa.mantBits)
            case (ia: IntTensor, ib: IntTensor) =>
              val out = ia.asInts.zip(ib.asInts).map { case (x, y) => x + y }
              nextTensor = IntTensor(curShape, out, ia.bitWidth)
            case _ =>
              throw new IllegalArgumentException("Cannot Add different tensor types")
          }

        case cc: Concat =>
          val ta = nodeOutputs(cc.a)
          val tb = nodeOutputs(cc.b)
          (ta, tb) match {
            case (fa: FloatTensor, fb: FloatTensor) =>
              val out = LayerReplicas.concat(fa.asFloats, fb.asFloats)
              nextShape = Seq(fa.shape.head + fb.shape.head) ++ fa.shape.tail
              nextTensor = FloatTensor(nextShape, out, fa.expBits, fa.mantBits)
            case (ia: IntTensor, ib: IntTensor) =>
              val out = ia.asInts ++ ib.asInts
              nextShape = Seq(ia.shape.head + ib.shape.head) ++ ia.shape.tail
              nextTensor = IntTensor(nextShape, out, ia.bitWidth)
            case _ =>
              throw new IllegalArgumentException("Cannot Concat different tensor types")
          }

        case _: Softmax =>
          // Pass-through for logits comparison
          nextTensor = curTensor

        case other =>
          println(s"[Warning] Layer $other skipped in replica trace.")
      }

      val doubles = nextTensor.toDoubles
      val first3 = doubles.take(3)
      val last3 = doubles.takeRight(3)
      val minVal = if (doubles.nonEmpty) doubles.min else 0.0
      val maxVal = if (doubles.nonEmpty) doubles.max else 0.0

      println(f"Layer $i%2d [$lName%-10s] : shape ${nextShape.mkString("x")}%-12s | first3=${formatPreview(first3)} ... last3=${formatPreview(last3)} | min=$minVal%6.3f, max=$maxVal%6.3f")

      val trace = LayerExecutionTrace(
        layerIdx = i,
        layerName = lName,
        outShape = nextShape,
        first3 = first3,
        last3 = last3,
        minVal = minVal,
        maxVal = maxVal,
        tensor = if (nextTensor.isInstanceOf[FloatTensor]) nextTensor.asFloats else Nil,
        replicaTensor = Some(nextTensor)
      )

      traces += trace
      nodeOutputs += nextTensor
      curShape = nextShape
      curTensor = nextTensor
    }

    println("=" * 80 + "\n")

    val finalLogits = curTensor.toDoubles
    ForwardResult(finalLogits, traces.toSeq)
  }

  /**
   * Overloaded forwardWithTrace for backward compatibility with Seq[F] input.
   */
  def forwardWithTrace(
    layers: Seq[LayerSpec],
    inputShape: Seq[Int],
    inputValues: Seq[F],
    packedWeights: WeightMemoryLayout.PackedWeightsResult,
    expBits: Int = 8,
    mantBits: Int = 7
  ): ForwardResult = {
    forwardWithTrace(layers, inputShape, FloatTensor(inputShape, inputValues, expBits, mantBits), packedWeights)
  }
}
