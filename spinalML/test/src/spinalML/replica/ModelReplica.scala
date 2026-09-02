package spinalML.replica

import scala.collection.mutable.ArrayBuffer
import spinalML.nn._
import HWArithmetic._

/**
 * Universal interpreter of Seq[LayerSpec] computing the bit-exact software oracle.
 */
object ModelReplica {

  /**
   * Evaluates an end-to-end model forward pass bit-exactly.
   * 
   * @param layers    LayerSpec definitions from the model
   * @param input2D   Input image as channels x height x width (in F field triples)
   * @param weightsMap Mapping of layer index to weights/biases
   */
  def forwardBF16(
    layers: Seq[LayerSpec],
    input: Array[Array[Array[F]]],
    convWeights: Seq[Seq[Seq[F]]],
    convBiases: Seq[Seq[F]],
    linearWeights: Seq[Seq[Seq[F]]],
    linearBiases: Seq[Seq[F]]
  ): Seq[Double] = {
    val eb = 8 // BF16
    val mb = 7

    var cur2D = input
    var cur1D: Seq[F] = null
    var isFlat = false

    var convIdx = 0
    var linearIdx = 0

    for (layer <- layers) {
      layer match {
        case c: Conv2D =>
          cur2D = LayerReplicas.conv2D(
            cur2D,
            convWeights(convIdx),
            convBiases(convIdx),
            c.inChannels,
            c.outChannels,
            c.kernelSize,
            eb,
            mb
          )
          convIdx += 1

        case _: ReLU =>
          if (!isFlat) {
            cur2D = LayerReplicas.relu(cur2D)
          } else {
            cur1D = cur1D.map(v => if (v.s) PZERO else v)
          }

        case p: MaxPool2D =>
          cur2D = LayerReplicas.maxPool2D(cur2D, p.poolSize, p.stride, eb, mb)

        case _: Flatten =>
          cur1D = LayerReplicas.flatten(cur2D)
          isFlat = true

        case l: Linear =>
          val lanes = l.effLanes
          cur1D = LayerReplicas.linear(
            cur1D,
            linearWeights(linearIdx),
            linearBiases(linearIdx),
            eb,
            mb,
            lanes
          )
          linearIdx += 1

        case other =>
          throw new IllegalArgumentException(s"Layer $other not yet implemented in ModelReplica forward pass")
      }
    }

    cur1D.map(f => decode(f, eb, mb))
  }
}
