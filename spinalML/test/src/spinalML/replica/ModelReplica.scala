package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.nn._
import HWArithmetic._
import handlers._

/**
 * Universal interpreter of Seq[LayerSpec] computing the bit-exact software oracle
 * and printing intermediate activation previews (first 3 and last 3 elements, min/max).
 *
 * Refactored into modular layer handlers to preserve low cyclomatic complexity (cc < 15)
 * and enforce strict architectural separation of concerns.
 */
object ModelReplica {

  // Re-export core types for full backward compatibility
  type ReplicaTensor = spinalML.replica.ReplicaTensor
  type IntTensor = spinalML.replica.IntTensor
  val IntTensor = spinalML.replica.IntTensor
  type FloatTensor = spinalML.replica.FloatTensor
  val FloatTensor = spinalML.replica.FloatTensor
  type LayerExecutionTrace = spinalML.replica.LayerExecutionTrace
  val LayerExecutionTrace = spinalML.replica.LayerExecutionTrace
  type ForwardResult = spinalML.replica.ForwardResult
  val ForwardResult = spinalML.replica.ForwardResult

  private def formatPreview(s: Seq[Double]): String = {
    s.map(v => f"$v%6.3f").mkString("[", ", ", "]")
  }

  /**
   * Dispatches execution of a single layer to its specialized modular handler.
   */
  private def dispatchLayer(
    layer: LayerSpec,
    curTensor: ReplicaTensor,
    curShape: Seq[Int],
    wInfo: WeightMemoryLayout.LayerWeightInfo,
    nodeOutputs: Seq[ReplicaTensor]
  ): (Seq[Int], ReplicaTensor) = layer match {
    case c: Conv2D       => ConvHandlers.evalConv2D(c, curTensor, curShape, wInfo)
    case c: Conv1D       => ConvHandlers.evalConv1D(c, curTensor, curShape, wInfo)
    case l: Linear       => DenseHandlers.evalLinear(l, curTensor, curShape, wInfo)
    case bn: BatchNorm1D => DenseHandlers.evalBatchNorm1D(bn, curTensor, curShape, wInfo)
    case p: MaxPool2D    => PoolHandlers.evalMaxPool2D(p, curTensor, curShape)
    case p: MaxPool1D    => PoolHandlers.evalMaxPool1D(p, curTensor, curShape)
    case _: ReLU         => ActivationHandlers.evalReLU(curTensor, curShape)
    case lr: LeakyReLU   => ActivationHandlers.evalLeakyReLU(lr, curTensor, curShape)
    case c: Cast         => TransformHandlers.evalCast(c, curTensor, curShape)
    case _: Flatten      => TransformHandlers.evalFlatten(curTensor, curShape)
    case ad: Add         => DagHandlers.evalAdd(ad, curShape, nodeOutputs)
    case cc: Concat      => DagHandlers.evalConcat(cc, curShape, nodeOutputs)
    case _: Softmax      => (curShape, curTensor) // Pass-through for logits comparison
    case other           =>
      println(s"[Warning] Layer $other skipped in replica trace.")
      (curShape, curTensor)
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

      val (nextShape, nextTensor) = dispatchLayer(layer, curTensor, curShape, wInfo, nodeOutputs)

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
