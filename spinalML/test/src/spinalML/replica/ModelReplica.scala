package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.nn._
import HWArithmetic._

/**
 * Universal interpreter of Seq[LayerSpec] computing the bit-exact software oracle
 * and printing intermediate activation previews (first 3 and last 3 elements, min/max).
 */
object ModelReplica {

  case class LayerExecutionTrace(
    layerIdx: Int,
    layerName: String,
    outShape: Seq[Int],
    first3: Seq[Double],
    last3: Seq[Double],
    minVal: Double,
    maxVal: Double,
    tensor: Seq[F]
  )

  case class ForwardResult(
    logits: Seq[Double],
    traces: Seq[LayerExecutionTrace]
  )

  private def formatPreview(s: Seq[Double]): String = {
    s.map(v => f"$v%6.3f").mkString("[", ", ", "]")
  }

  /**
   * Universal forward pass execution on any Accelerator / Sequential model.
   */
  def forwardWithTrace(
    layers: Seq[LayerSpec],
    inputShape: Seq[Int],
    inputValues: Seq[F],
    packedWeights: WeightMemoryLayout.PackedWeightsResult,
    expBits: Int = 8,
    mantBits: Int = 7
  ): ForwardResult = {

    println("\n" + "=" * 80)
    println("Model Architecture & Intermediate Activation Trace:")
    println("-" * 80)

    val inputDoubles = inputValues.map(f => decode(f, expBits, mantBits))
    val inFirst3 = inputDoubles.take(3)
    val inLast3 = inputDoubles.takeRight(3)
    val inMin = if (inputDoubles.nonEmpty) inputDoubles.min else 0.0
    val inMax = if (inputDoubles.nonEmpty) inputDoubles.max else 0.0

    println(f"Input Tensor        : shape ${inputShape.mkString("x")}%-12s | first3=${formatPreview(inFirst3)} ... last3=${formatPreview(inLast3)} | min=$inMin%6.3f, max=$inMax%6.3f")

    // Node storage for DAG execution
    val nodeOutputs = ArrayBuffer[Seq[F]]()
    val traces = ArrayBuffer[LayerExecutionTrace]()

    nodeOutputs += inputValues

    var curShape = inputShape
    var curTensor = inputValues

    for (i <- layers.indices) {
      val layer = layers(i)
      val wInfo = packedWeights.layers(i)
      val lName = layer.getClass.getSimpleName

      var nextShape = curShape
      var nextTensor = curTensor

      layer match {
        case c: Conv2D =>
          require(curShape.length >= 2, "Conv2D requires 2D or 3D input shape")
          val h = curShape(0)
          val w = curShape(1)
          val inC = if (curShape.length >= 3) curShape(2) else 1
          
          // Reshape flat into [C][H][W]
          val arr3D = Array.ofDim[F](inC, h, w)
          var idx = 0
          for (y <- 0 until h; x <- 0 until w; ch <- 0 until inC) {
            arr3D(ch)(y)(x) = if (idx < curTensor.length) curTensor(idx) else PZERO
            idx += 1
          }

          val kSize = c.kernelSize
          val outC = c.outChannels
          val kElems = kSize * kSize * inC
          val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
          val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)

          val convOut = LayerReplicas.conv2D(arr3D, convW, convB, inC, outC, kSize, expBits, mantBits)
          val hOut = convOut(0).length
          val wOut = convOut(0)(0).length
          nextShape = Seq(hOut, wOut, outC)

          // Flatten back to features-last
          nextTensor = LayerReplicas.flatten(convOut)

        case c: Conv1D =>
          val l = curShape(0)
          val inC = if (curShape.length >= 2) curShape(1) else 1
          val arr2D = Array.ofDim[F](l, inC)
          var idx = 0
          for (pos <- 0 until l; ch <- 0 until inC) {
            arr2D(pos)(ch) = if (idx < curTensor.length) curTensor(idx) else PZERO
            idx += 1
          }

          val kSize = c.kernelSize
          val outC = c.outChannels
          val kElems = kSize * inC
          val convW = (0 until outC).map(o => wInfo.weightValues.slice(o * kElems, (o + 1) * kElems))
          val convB = (0 until outC).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)

          val convOut = LayerReplicas.conv1D(arr2D, convW, convB, inC, outC, kSize, expBits, mantBits)
          nextShape = Seq(convOut.length, outC)
          val flat = ArrayBuffer[F]()
          for (pos <- convOut.indices; ch <- 0 until outC) flat += convOut(pos)(ch)
          nextTensor = flat.toSeq

        case _: ReLU =>
          nextTensor = LayerReplicas.relu1D(curTensor)

        case lr: LeakyReLU =>
          nextTensor = LayerReplicas.leakyRelu(curTensor, lr.shift, expBits, mantBits)

        case p: MaxPool2D =>
          val h = curShape(0); val w = curShape(1); val c = if (curShape.length >= 3) curShape(2) else 1
          val arr3D = Array.ofDim[F](c, h, w)
          var idx = 0
          for (y <- 0 until h; x <- 0 until w; ch <- 0 until c) {
            arr3D(ch)(y)(x) = if (idx < curTensor.length) curTensor(idx) else PZERO
            idx += 1
          }
          val pooled = LayerReplicas.maxPool2D(arr3D, p.poolSize, p.stride, expBits, mantBits)
          nextShape = Seq(pooled(0).length, pooled(0)(0).length, c)
          nextTensor = LayerReplicas.flatten(pooled)

        case p: MaxPool1D =>
          val l = curShape(0); val c = if (curShape.length >= 2) curShape(1) else 1
          val arr2D = Array.ofDim[F](l, c)
          var idx = 0
          for (pos <- 0 until l; ch <- 0 until c) {
            arr2D(pos)(ch) = if (idx < curTensor.length) curTensor(idx) else PZERO
            idx += 1
          }
          val pooled = LayerReplicas.maxPool1D(arr2D, p.poolSize, p.stride, expBits, mantBits)
          nextShape = Seq(pooled.length, c)
          val flat = ArrayBuffer[F]()
          for (pos <- pooled.indices; ch <- 0 until c) flat += pooled(pos)(ch)
          nextTensor = flat.toSeq

        case _: Flatten =>
          nextShape = Seq(1, curShape.product)
          nextTensor = curTensor

        case l: Linear =>
          val inFeatures = l.inFeatures
          val outFeatures = l.outFeatures
          val lanes = l.effLanes
          val fcW = (0 until outFeatures).map(o => wInfo.weightValues.slice(o * inFeatures, (o + 1) * inFeatures))
          val fcB = (0 until outFeatures).map(o => if (o < wInfo.biasValues.length) wInfo.biasValues(o) else PZERO)

          nextTensor = LayerReplicas.linear(curTensor, fcW, fcB, expBits, mantBits, lanes)
          nextShape = Seq(1, outFeatures)

        case bn: BatchNorm1D =>
          val feat = bn.features
          val gamma = wInfo.weightValues.take(feat)
          val beta = wInfo.biasValues.take(feat)
          nextTensor = LayerReplicas.batchNorm1D(curTensor, gamma, beta, expBits, mantBits)

        case ad: Add =>
          val ta = nodeOutputs(ad.a)
          val tb = nodeOutputs(ad.b)
          nextTensor = LayerReplicas.add(ta, tb, expBits, mantBits)

        case cc: Concat =>
          val ta = nodeOutputs(cc.a)
          val tb = nodeOutputs(cc.b)
          nextTensor = LayerReplicas.concat(ta, tb)

        case _: Softmax =>
          // Pass-through for logits comparison
          nextTensor = curTensor

        case other =>
          println(s"[Warning] Layer $other skipped in replica trace.")
      }

      val doubles = nextTensor.map(f => decode(f, expBits, mantBits))
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
        tensor = nextTensor
      )

      traces += trace
      nodeOutputs += nextTensor
      curShape = nextShape
      curTensor = nextTensor
    }

    println("=" * 80 + "\n")

    val finalLogits = curTensor.map(f => decode(f, expBits, mantBits))
    ForwardResult(finalLogits, traces.toSeq)
  }
}
