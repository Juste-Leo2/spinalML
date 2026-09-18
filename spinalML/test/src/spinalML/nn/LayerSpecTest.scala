// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinalML.dtypes.{BF16, I8, I32}
import spinalML.layers.{LinearLayer, Conv1DLayer, Conv2DLayer,
  BatchNorm1D => BatchNorm1DHW, LayerNorm1D => LayerNorm1DHW}
import spinalML.attention.{ClassicalAttention, ClassicalAttentionHW}

/**
 * LAY-02 generic guard: elaborate one hardware instance per weight-carrying
 * `LayerSpec` family with tiny shapes and check that the ports Sequential
 * feeds (`io.w`/`io.b`, `gamma`/`beta`, `io.y`) match the declarative
 * metadata. A divergence means the DMA region layout (getWeightShape.product),
 * the bias region or the output-shape inference disagrees with the RTL — the
 * exact defect class LAY-02 belonged to (2D norm metadata vs 1D ports).
 *
 * The checks are deliberately non-tautological wherever the RTL rebuilds the
 * shape from independent parameters (Conv K/inC/outC, attention slicing),
 * and pin the derived relations the layers compute themselves (Linear bias
 * [1,N], attention y from embedDim).
 */
case class LayerSpecConformanceComp[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc]
) extends Component {

  private def check(label: String, observed: Seq[Int], declared: Seq[Int]): Unit =
    require(observed == declared,
      s"LAY-02 conformance: $label hardware port $observed != LayerSpec $declared")

  // Linear: the weight shape is passed through, but the bias port is rebuilt
  // from its column count ([1, N]) and the output from shapeA(0).
  {
    val spec = Linear(inFeatures = 4, outFeatures = 3)
    val hw = LinearLayer(dataType, dataType, accType, Seq(1, spec.inFeatures),
      spec.getWeightShape(), lanes = spec.effLanes)
    check("Linear weight", hw.io.w.shape, spec.getWeightShape())
    check("Linear bias", hw.io.b.shape, spec.getBiasShape())
    check("Linear output", hw.io.y.shape, spec.getOutShape(Seq(1, spec.inFeatures)))
  }

  // Conv1D/Conv2D: ports are rebuilt from (K, inChannels, outChannels).
  {
    val spec = Conv1D(inChannels = 2, outChannels = 3, kernelSize = 2)
    val hw = Conv1DLayer(dataType, accType, L_in = 4, inChannels = spec.inChannels,
      outChannels = spec.outChannels, K = spec.kernelSize, outLanes = 1,
      tileSize = spec.kernelSize * spec.inChannels)
    check("Conv1D weight", hw.io.w.shape, spec.getWeightShape())
    check("Conv1D bias", hw.io.b.shape, spec.getBiasShape())
    check("Conv1D output", hw.io.y.shape, spec.getOutShape(Seq(4, spec.inChannels)))
  }
  {
    val spec = Conv2D(inChannels = 2, outChannels = 3, kernelSize = 2)
    val hw = Conv2DLayer(dataType, accType, H = 4, W_in = 4, inChannels = spec.inChannels,
      outChannels = spec.outChannels, K = spec.kernelSize, outLanes = 1,
      tileSize = spec.kernelSize * spec.kernelSize * spec.inChannels)
    check("Conv2D weight", hw.io.w.shape, spec.getWeightShape())
    check("Conv2D bias", hw.io.b.shape, spec.getBiasShape())
    check("Conv2D output", hw.io.y.shape, spec.getOutShape(Seq(4, 4, spec.inChannels)))
  }

  // Norms: the LAY-02 regression sent 2D metadata to 1D gamma/beta ports.
  {
    val spec = BatchNorm1D(features = 4)
    val hw = BatchNorm1DHW(dataType, channels = spec.features, seqLen = 2)
    check("BatchNorm1D gamma", hw.io.gamma.shape, spec.getWeightShape())
    check("BatchNorm1D beta", hw.io.beta.shape, spec.getBiasShape())
    check("BatchNorm1D output", hw.io.y.shape, spec.getOutShape(Seq(2, spec.features)))
  }
  {
    val spec = LayerNorm1D(features = 4)
    val hw = LayerNorm1DHW(dataType, channels = spec.features, seqLen = 2)
    check("LayerNorm1D gamma", hw.io.gamma.shape, spec.getWeightShape())
    check("LayerNorm1D beta", hw.io.beta.shape, spec.getBiasShape())
    check("LayerNorm1D output", hw.io.y.shape, spec.getOutShape(Seq(2, spec.features)))
  }

  // Attention: one DMA region [4*embedDim, embedDim] sliced by Sequential into
  // four [embedDim, embedDim] projection ports, and no bias region at all.
  {
    val spec = ClassicalAttention(embedDim = 4)
    val hw = ClassicalAttentionHW(dataType, dataType, accType, seqLen = 2,
      embedDim = spec.embedDim, numHeads = spec.numHeads, xLanes = 1, wLanes = spec.embedDim)
    val ports = Seq(hw.io.wq, hw.io.wk, hw.io.wv, hw.io.wo)
    val portRows = ports.map(_.shape.head).sum
    require(portRows == spec.getWeightShape().head,
      s"LAY-02 conformance: ClassicalAttention hardware rows $portRows != LayerSpec ${spec.getWeightShape().head}")
    for ((p, i) <- ports.zipWithIndex) {
      check(s"ClassicalAttention weight port $i", p.shape.tail, spec.getWeightShape().tail)
    }
    require(spec.getBiasShape() == Seq(0),
      s"LAY-02 conformance: ClassicalAttention declares bias ${spec.getBiasShape()} but the hardware has no bias port")
    check("ClassicalAttention output", hw.io.y.shape, spec.getOutShape(Seq(2, spec.embedDim)))
  }
}

class LayerSpecTest extends AnyFunSuite {

  test("Linear getOutShape preserves leading rows (features-last)") {
    assert(Linear(4, 3).getOutShape(Seq(2, 4)) == Seq(2, 3))
    assert(Linear(64, 10).getOutShape(Seq(1, 64)) == Seq(1, 10))
    assert(Linear(64, 10).getOutShape(Seq(3, 2, 64)) == Seq(3, 2, 10))
    assert(Linear(8, 4).getOutShape(Seq(5, 6, 8)) == Seq(5, 6, 4))
  }

  test("Linear getOutShape rejects mismatched feature dimension") {
    intercept[IllegalArgumentException](Linear(5, 3).getOutShape(Seq(2, 4)))
    intercept[IllegalArgumentException](Linear(5, 3).getOutShape(Seq(7)))
  }

  test("Flatten produces features-last vector") {
    assert(Flatten().getOutShape(Seq(8, 8)) == Seq(1, 64))
    assert(Flatten().getOutShape(Seq(6, 6, 1)) == Seq(1, 36))
    assert(Flatten().getOutShape(Seq(4)) == Seq(1, 4))
  }

  test("Flatten -> Linear chain shapes") {
    val shape = Flatten().getOutShape(Seq(6, 6, 4))
    assert(shape == Seq(1, 144))
    assert(Linear(144, 10).getOutShape(shape) == Seq(1, 10))
  }

  test("Shape-only specs are identity") {
    for (shape <- Seq(Seq(4), Seq(2, 3), Seq(2, 3, 4))) {
      assert(ReLU().getOutShape(shape) == shape)
      assert(Sigmoid().getOutShape(shape) == shape)
      assert(Tanh().getOutShape(shape) == shape)
      assert(Cast(BF16()).getOutShape(shape) == shape)
    }
  }

  test("BatchNorm1D/LayerNorm1D weight and bias shapes are 1D (LAY-02)") {
    // The hardware ports are Tensor(dataType, Seq(channels)) (batchnorm.scala /
    // layernorm.scala): the declarative metadata must match, otherwise shape
    // checks and the generic LayerSpec compliance test disagree with the RTL.
    for (spec <- Seq[LayerSpec](BatchNorm1D(4), LayerNorm1D(4))) {
      assert(spec.getWeightShape() == Seq(4), s"$spec weights")
      assert(spec.getBiasShape() == Seq(4), s"$spec bias")
    }
  }

  test("LayerSpec metadata matches hardware ports (LAY-02)") {
    // The wrapper's require()s fire during elaboration; any mismatch
    // propagates out of generateVerilog and fails this test.
    SpinalConfig().generateVerilog(LayerSpecConformanceComp(I8(), I32()))
  }

  test("weightless LayerSpecs declare no weight/bias region (LAY-02)") {
    // Sequential.scala:204 counts one DMA trigger per spec whose
    // getWeightShape().head > 0: a stray non-zero shape would fabricate a
    // weight/bias region the hardware never consumes.
    val weightless: Seq[LayerSpec] = Seq(
      ReLU(), LeakyReLU(), Softmax(), Sigmoid(), Tanh(),
      MaxPool1D(2, 2), AvgPool1D(2, 2), MaxPool2D(2, 2), AvgPool2D(2, 2),
      Flatten(), Cast(BF16()), Requantize(1, BF16()), Repack(2),
      Add(0, 1), Concat(0, 1)
    )
    for (spec <- weightless) {
      assert(spec.getWeightShape() == Seq(0), s"$spec weight shape")
      assert(spec.getBiasShape() == Seq(0), s"$spec bias shape")
    }
  }
}
