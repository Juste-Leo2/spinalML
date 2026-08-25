package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.tensors.Tensor
import spinalML.utils.MemLayout
import spinalML.dtypes.FloatML
import spinalML.layers.{Conv1D => Conv1DHW, Conv2D => Conv2DHW, Linear => LinearHW, batchnorm}
import spinalML.ops.{reshape, repack, flatten, cast}
import spinalML.activations.{relu, leaky_relu, sigmoid, tanh}
import spinalML.poolings.{maxpool1d, avgpool1d, maxpool2d, avgpool2d}
import spinalML.attention._

case class Sequential(
  globalDataType: HardType[Data],
  inputShape: Seq[Int],
  layers: Seq[LayerSpec],
  axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Component {

  // ============================================================
  // 0. Topology analysis (pure elaboration-time, no hardware yet)
  // ============================================================
  // Node 0 is the network input; node k (k >= 1) is the output of layers(k - 1).
  // Standard layers implicitly consume the previous node (index = their position);
  // merge layers (Add / Concat) consume two explicitly referenced earlier nodes,
  // which keeps the graph acyclic by construction.
  val nNodes = layers.length + 1

  private def consumedNodes(layerIdx: Int): Seq[Int] = layers(layerIdx) match {
    case ad: Add    => Seq(ad.a, ad.b)
    case cc: Concat => Seq(cc.a, cc.b)
    case _          => Seq(layerIdx)
  }

  private def sameDtype(x: HardType[Data], y: HardType[Data]): Boolean = {
    val dx = x(); val dy = y()
    dx.getClass == dy.getClass && dx.getBitsWidth == dy.getBitsWidth
  }

  val consumers = Array.fill(nNodes)(scala.collection.mutable.ArrayBuffer[Int]())
  for (i <- layers.indices; n <- consumedNodes(i)) {
    require(n <= i,
      s"Layer $i (${layers(i)}) references node $n: DAG references must point to earlier nodes [0..$i]")
    consumers(n) += i
  }
  consumers(nNodes - 1) += -1 // -1 marks the accelerator output

  for (n <- 0 until nNodes) {
    require(consumers(n).nonEmpty,
      s"Node $n (${if (n == 0) "network input" else s"output of ${layers(n - 1)}"}) is never consumed")
  }

  val nodeShapes = scala.collection.mutable.ArrayBuffer[Seq[Int]](inputShape)
  val nodeTypes = scala.collection.mutable.ArrayBuffer[HardType[Data]](globalDataType)

  for (i <- layers.indices) {
    val l = layers(i)
    val outShape = l match {
      case ad: Add =>
        val sa = nodeShapes(ad.a); val sb = nodeShapes(ad.b)
        require(sa == sb,
          s"Add(${ad.a}, ${ad.b}) requires identical shapes: $sa vs $sb")
        sa
      case cc: Concat =>
        val sa = nodeShapes(cc.a); val sb = nodeShapes(cc.b)
        require(sa.length == sb.length && sa.tail == sb.tail,
          s"Concat(${cc.a}, ${cc.b}, axis=0) requires identical tail shapes: $sa vs $sb")
        sa.updated(0, sa.head + sb.head)
      case _ => l.getOutShape(nodeShapes(i))
    }
    val outType = l match {
      case ad: Add =>
        require(sameDtype(nodeTypes(ad.a), nodeTypes(ad.b)),
          s"Add(${ad.a}, ${ad.b}) requires identical dtypes: ${nodeTypes(ad.a)} vs ${nodeTypes(ad.b)} (insert a Cast layer on one branch)")
        nodeTypes(ad.a)
      case cc: Concat =>
        require(sameDtype(nodeTypes(cc.a), nodeTypes(cc.b)),
          s"Concat(${cc.a}, ${cc.b}) requires identical dtypes: ${nodeTypes(cc.a)} vs ${nodeTypes(cc.b)} (insert a Cast layer on one branch)")
        nodeTypes(cc.a)
      case _ => l.outType(nodeTypes(i))
    }
    nodeShapes += outShape
    nodeTypes += outType
  }

  def computeFinalShape(): (Seq[Int], HardType[Data]) = (nodeShapes.last, nodeTypes.last)
  val (finalShape, finalType) = computeFinalShape()

  val io = new Bundle {
    val start = slave(Event)
    val imgBaseAddress = in UInt(axiConfig.addressWidth bits)
    val weightsBaseAddress = in UInt(axiConfig.addressWidth bits)

    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(finalType, finalShape, lanes = 1)) // Default to 1 lane output for now
  }

  // --- 1. Memory Offset Calculation & DMA Instantiation ---
  val allAxiMasters = scala.collection.mutable.ArrayBuffer[Axi4ReadOnly]()
  var currentMemoryOffset = 0

  // Every weight/bias region must start on an AXI-beat boundary: DDR
  // controllers and memory models serve bursts from the beat-aligned address,
  // so an unaligned region start would silently read the tail of the previous
  // region's word instead of the intended first elements.
  val beatBytes = axiConfig.dataWidth / 8
  def alignToBeat(offset: Int): Int = MemLayout.alignToBeat(offset, beatBytes)

  // Start triggers fork
  // Image + each weight + each bias
  val totalDmaTriggers = 1 + layers.map(l => (if(l.getWeightShape().head > 0) 1 else 0) + (if(l.getBiasShape().head > 0) 1 else 0)).sum
  val startTriggers = StreamFork(io.start, totalDmaTriggers)
  var triggerIdx = 0

  // Base AXI config for leaf DMAs (accounting for arbiter routing bits)
  // For simplicity in this V1, we use a single stage arbiter if <= 16 ports.
  // A single DMA (weightless models) skips the arbiter entirely and keeps the
  // full id width.
  val dmaAxiConfig =
    if (totalDmaTriggers == 1) axiConfig
    else axiConfig.copy(idWidth = axiConfig.idWidth - log2Up(totalDmaTriggers))

  // 1.1. Image DMA
  val inputDataType = globalDataType
  val dmaImg = DMAReader2D(inputDataType, inputShape, outLanes = 1, dmaAxiConfig)
  dmaImg.io.cmd.valid := startTriggers(triggerIdx).valid
  startTriggers(triggerIdx).ready := dmaImg.io.cmd.ready
  dmaImg.io.cmd.baseAddress := io.imgBaseAddress

  // Stride is typically width * byteSize
  val pixelBytes = inputDataType.getBitsWidth / 8
  dmaImg.io.cmd.stride := inputShape(1) * pixelBytes
  val elementsPerBeat = axiConfig.dataWidth / inputDataType.getBitsWidth
  require(elementsPerBeat >= 1,
    s"Input dtype (${inputDataType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported image element size")
  dmaImg.io.cmd.patchWidth := (scala.math.ceil(inputShape(1).toDouble / elementsPerBeat).toInt - 1).max(0)
  dmaImg.io.cmd.patchHeight := inputShape(0)

  allAxiMasters += dmaImg.io.axiMaster
  triggerIdx += 1

  // Use Automatic Double Buffering instead of a simple queue
  // The double-buffer contract tiles exactly `depth` elements per bank: the
  // buffer MUST be sized to the exact tensor size (any floor like max(16, n)
  // deadlocks small tensors, as tileReady would never assert).
  val imgBufferSize = inputShape.product
  val imgDoubleBuffer = StreamDoubleBuffer(inputDataType, imgBufferSize, lanes = 1)
  imgDoubleBuffer.io.streamIn << dmaImg.io.outStream.stream

  val imgStreamer = DoubleBufferStreamer(inputDataType, imgBufferSize, lanes = 1)
  imgStreamer.io.readData := imgDoubleBuffer.io.readData
  imgStreamer.io.tileReady := imgDoubleBuffer.io.tileReady
  imgDoubleBuffer.io.readAddr := imgStreamer.io.readAddr
  imgDoubleBuffer.io.nextTile := imgStreamer.io.nextTile

  val imgQueue = Tensor(inputDataType, inputShape, 1)
  imgQueue.stream << imgStreamer.io.streamOut

  // --- 2. Node production ---
  // Every produced tensor becomes a graph node. Nodes consumed more than once
  // are forked; deferred branches flow through exact-capacity TapBuffers so a
  // one-shot inference never overflows them.
  val nodeOutputs = scala.collection.mutable.ArrayBuffer[scala.collection.mutable.ArrayBuffer[Tensor[Data]]]()

  def registerNode(raw: Tensor[Data]): Unit = {
    val cons = consumers(nodeOutputs.length)
    val views = if (cons.length > 1) TapBuffer.fork(raw, cons.length) else Seq(raw)
    nodeOutputs += scala.collection.mutable.ArrayBuffer(views: _*)
  }

  def inputFor(ref: Int, layerIdx: Int): Tensor[Data] =
    nodeOutputs(ref)(consumers(ref).indexOf(layerIdx))

  registerNode(imgQueue)

  for (i <- layers.indices) {
    val layer = layers(i)
    val lType = layer.outType(nodeTypes(i))
    val wType = layer.weightType(nodeTypes(i))

    val wShape = layer.getWeightShape()
    val bShape = layer.getBiasShape()

    var layerWeights: Tensor[Data] = null
    var layerBias: Tensor[Data] = null

    // Fetch Weights
    if (wShape.head > 0) {
      val elements = wShape.product
      // For Linear, lanes might be very high. We should cap it or repack. For this V1, we assume small kernel/linear.
      // Wait, Conv2D requires w.lanes == K*K.
      // Linear requires w.lanes == inFeatures.
      val requiredLanes = layer match {
        case c: Conv2D => c.kernelSize * c.kernelSize
        case c: Conv1D => c.kernelSize * c.inChannels
        case l: Linear => l.inFeatures
        case bn: BatchNorm1D => bn.features
        case ln: LayerNorm1D => ln.features
        case a: ClassicalAttention => a.embedDim
        case _ => 1
      }

      val dmaW = DMAReader(wType, wShape, outLanes = requiredLanes, dmaAxiConfig)
      val reqW = Stream(FetchRequest(axiConfig.addressWidth))
      reqW.valid := startTriggers(triggerIdx).valid
      startTriggers(triggerIdx).ready := reqW.ready
      currentMemoryOffset = alignToBeat(currentMemoryOffset)
      reqW.address := io.weightsBaseAddress + currentMemoryOffset
      val elementsPerBeatW = axiConfig.dataWidth / wType.getBitsWidth
      require(elementsPerBeatW >= 1,
        s"Weight dtype (${wType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported weight element size")
      val beats = (elements + elementsPerBeatW - 1) / elementsPerBeatW
      require(beats <= 65536,
        s"Weight region of layer $i needs $beats beats, above the 64K-beat single-command limit (multi-command fetch belongs to the multi-tile roadmap)")
      reqW.length := (if (beats > 0) beats - 1 else 0)
      dmaW.io.cmd << reqW

      allAxiMasters += dmaW.io.axiMaster
      triggerIdx += 1
      // Whole-region ceil via MemLayout: sub-byte weight types (e.g. I4
      // nibbles) must not floor to zero bytes, or the next region aliases
      // the weight region start.
      currentMemoryOffset = alignToBeat(currentMemoryOffset + MemLayout.regionBytes(elements, wType.getBitsWidth))

      val wBufferSize = elements // Double buffer size for weights (exact: contract = tile of `depth` elements)
      val wDoubleBuffer = StreamDoubleBuffer(wType, wBufferSize, requiredLanes)
      wDoubleBuffer.io.streamIn << dmaW.io.outStream.stream

      val wStreamer = DoubleBufferStreamer(wType, wBufferSize, requiredLanes)
      wStreamer.io.readData := wDoubleBuffer.io.readData
      wStreamer.io.tileReady := wDoubleBuffer.io.tileReady
      wDoubleBuffer.io.readAddr := wStreamer.io.readAddr
      wDoubleBuffer.io.nextTile := wStreamer.io.nextTile

      layerWeights = Tensor(wType, wShape, requiredLanes)
      layerWeights.stream << wStreamer.io.streamOut
    }

    // Fetch Bias
    if (bShape.head > 0) {
      val requiredBiasLanes = layer match {
        case bn: BatchNorm1D => bn.features
        case ln: LayerNorm1D => ln.features
        case _ => 1
      }
      val elements = bShape.product
      val dmaB = DMAReader(lType, bShape, outLanes = requiredBiasLanes, dmaAxiConfig)
      val reqB = Stream(FetchRequest(axiConfig.addressWidth))
      reqB.valid := startTriggers(triggerIdx).valid
      startTriggers(triggerIdx).ready := reqB.ready
      currentMemoryOffset = alignToBeat(currentMemoryOffset)
      reqB.address := io.weightsBaseAddress + currentMemoryOffset

      val elementsPerBeatB = axiConfig.dataWidth / lType.getBitsWidth
      require(elementsPerBeatB >= 1,
        s"Bias dtype (${lType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported bias element size")
      val beatsB = (elements + elementsPerBeatB - 1) / elementsPerBeatB
      require(beatsB <= 65536,
        s"Bias region of layer $i needs $beatsB beats, above the 64K-beat single-command limit (multi-command fetch belongs to the multi-tile roadmap)")
      reqB.length := (if (beatsB > 0) beatsB - 1 else 0)

      dmaB.io.cmd << reqB

      allAxiMasters += dmaB.io.axiMaster
      triggerIdx += 1
      currentMemoryOffset = alignToBeat(currentMemoryOffset + MemLayout.regionBytes(elements, lType.getBitsWidth))

      val bBufferSize = elements // Exact size (contract = tile of `depth` elements)
      val bDoubleBuffer = StreamDoubleBuffer(lType, bBufferSize, requiredBiasLanes)
      bDoubleBuffer.io.streamIn << dmaB.io.outStream.stream

      val bStreamer = DoubleBufferStreamer(lType, bBufferSize, requiredBiasLanes)
      bStreamer.io.readData := bDoubleBuffer.io.readData
      bStreamer.io.tileReady := bDoubleBuffer.io.tileReady
      bDoubleBuffer.io.readAddr := bStreamer.io.readAddr
      bDoubleBuffer.io.nextTile := bStreamer.io.nextTile

      layerBias = Tensor(lType, bShape, requiredBiasLanes)
      layerBias.stream << bStreamer.io.streamOut
    }

    // Instantiate computation block
    val inTensor = inputFor(i, i)

    val nextTensor: Tensor[Data] = layer match {
      case c: Conv1D =>
        Conv1DHW(inTensor, layerWeights, layerBias, lType)

      case c: Conv2D =>
        // Integer-domain convolutions: narrow SInt weights (e.g. true I4
        // nibbles fetched from DDR) are sign-extended to the activation width
        // so the single-width int matmul can consume them.
        val wDT = layerWeights.dataType()
        val aDT = inTensor.dataType()
        val wForConv =
          if (wDT.isInstanceOf[SInt] && aDT.isInstanceOf[SInt] && wDT.getBitsWidth != aDT.getBitsWidth) {
            require(wDT.getBitsWidth < aDT.getBitsWidth,
              s"Conv2D weights (${wDT.getBitsWidth}b) wider than activations (${aDT.getBitsWidth}b): " +
                "narrowing auto-cast refused — insert an explicit Cast layer or use wider activations")
            cast(layerWeights.asInstanceOf[Tensor[SInt]], inTensor.dataType)
          } else {
            require(!(aDT.isInstanceOf[FloatML] && wDT.isInstanceOf[SInt]),
              "Conv2D with float activations and integer weights is unsupported (float dequant is Linear-only): " +
                "quantize the activations or run this stage on integer activations")
            layerWeights
          }
        Conv2DHW(inTensor, wForConv, layerBias, lType)

      case _: ReLU =>
        relu(inTensor)

      case lr: LeakyReLU =>
        leaky_relu(inTensor, lr.shift)

      case _: Softmax =>
        val seqLen = nodeShapes(i)(0)
        val channels = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val comp = spinalML.activations.Softmax1D(nodeTypes(i), channels, seqLen)
        // Softmax1D consumes and produces lanes = channels; keep the lanes=1
        // invariant on both sides like the pooling 2D layers.
        comp.io.x <> repack(inTensor, channels)
        val smOut = comp.io.y
        if (smOut.lanes != 1) repack(smOut, 1) else smOut

      case bn: BatchNorm1D =>
        batchnorm(inTensor, layerWeights, layerBias)

      case ln: LayerNorm1D =>
        val seqLen = nodeShapes(i)(0)
        val channels = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val comp = spinalML.layers.LayerNorm1D(nodeTypes(i), channels, seqLen)
        comp.io.x <> inTensor
        comp.io.gamma <> layerWeights
        comp.io.beta <> layerBias
        comp.io.y

      case mp: MaxPool1D =>
        val c = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val repacked = if (inTensor.lanes != c) repack(inTensor, c) else inTensor
        maxpool1d(repacked, mp.poolSize, mp.stride)

      case ap: AvgPool1D =>
        val c = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val repacked = if (inTensor.lanes != c) repack(inTensor, c) else inTensor
        avgpool1d(repacked, ap.poolSize, ap.stride)

      case mp2: MaxPool2D =>
        // Pooling 2D consumes one element per beat (lanes = 1) and emits C lanes per beat.
        // Repack on both sides to preserve the lanes = 1 invariant between layers.
        val in = if (inTensor.lanes != 1) repack(inTensor, 1) else inTensor
        val pooled = maxpool2d(in, mp2.poolSize, mp2.stride)
        if (pooled.lanes != 1) repack(pooled, 1) else pooled

      case ap2: AvgPool2D =>
        val in = if (inTensor.lanes != 1) repack(inTensor, 1) else inTensor
        val pooled = avgpool2d(in, ap2.poolSize, ap2.stride)
        if (pooled.lanes != 1) repack(pooled, 1) else pooled

      case _: Sigmoid =>
        sigmoid(inTensor)

      case _: Tanh =>
        tanh(inTensor)

      case _: Cast =>
        cast(inTensor, lType, layers(i).asInstanceOf[Cast].scales)

      case _: Flatten =>
        reshape(flatten(inTensor), Seq(1, inTensor.shape.product))

      case l: Linear =>
        val rows = inTensor.shape.dropRight(1).product
        val reshaped = reshape(inTensor, Seq(rows, l.inFeatures))
        val repackedTensor = repack(reshaped, l.inFeatures)
        // Weight-only quantization (wXaY): SInt weights (I4/I8) + compile-time scale(s)
        // are dequantized to the activation float dtype inside the layer.
        layerWeights.dataType() match {
          case _: SInt =>
            spinalML.layers.Linear(repackedTensor, layerWeights.asInstanceOf[Tensor[SInt]], layerBias, lType, l.weightScales)
          case _ =>
            LinearHW(repackedTensor, layerWeights, layerBias, lType)
        }

      case rq: Requantize =>
        spinalML.ops.requantize(inTensor, rq.targetType, rq.shift)

      case rp: Repack =>
        repack(inTensor, rp.newLanes)

      case ad: Add =>
        val ta = inputFor(ad.a, i)
        val tbRaw = inputFor(ad.b, i)
        val tb = if (tbRaw.lanes != ta.lanes) repack(tbRaw, ta.lanes) else tbRaw
        spinalML.ops.add(ta, tb)

      case cc: Concat =>
        val ta = inputFor(cc.a, i)
        val tbRaw = inputFor(cc.b, i)
        val tb = if (tbRaw.lanes != ta.lanes) repack(tbRaw, ta.lanes) else tbRaw
        spinalML.ops.concatenate(ta, tb, 0)

      case a: ClassicalAttention =>
        val seqLen = nodeShapes(i)(0)
        // wType carries the (possibly quantized) weight dtype declared via
        // customWeightType; scales drive the in-layer dequantization.
        val comp = ClassicalAttentionHW(nodeTypes(i), wType, lType, seqLen, a.embedDim, a.numHeads, inTensor.lanes, layerWeights.lanes, weightScales = a.weightScales)
        comp.io.x <> inTensor

        // Fork and slice the weights stream into 4 parts
        val wForks = StreamFork(layerWeights.stream, 4)

        val w0 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w1 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w2 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w3 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        w0.stream << wForks(0)
        w1.stream << wForks(1)
        w2.stream << wForks(2)
        w3.stream << wForks(3)

        comp.io.wq <> spinalML.ops.slice(w0, 0, a.embedDim, axis = 0)
        comp.io.wk <> spinalML.ops.slice(w1, a.embedDim, 2 * a.embedDim, axis = 0)
        comp.io.wv <> spinalML.ops.slice(w2, 2 * a.embedDim, 3 * a.embedDim, axis = 0)
        comp.io.wo <> spinalML.ops.slice(w3, 3 * a.embedDim, 4 * a.embedDim, axis = 0)

        comp.io.y
    }

    registerNode(nextTensor)
  }

  // Output assignment: the last node feeds the accelerator output stream
  io.outStream <> nodeOutputs.last(consumers.last.indexOf(-1))

  // --- 3. Hierarchical AXI Arbitration ---
  // To avoid long combinatorial paths with many DMAs, we build a tree.
  // For V1, if port count <= 16, we just use one. Otherwise, tree.
  if (allAxiMasters.length <= 16) {
    val arbiter = Axi4ReadOnlyArbiter(axiConfig, allAxiMasters.length)
    for (i <- allAxiMasters.indices) {
      arbiter.io.inputs(i) <> allAxiMasters(i)
    }
    io.axiMaster <> arbiter.io.output
  } else {
    // Hierarchical tree logic (placeholder for future expansion, groups of 4)
    // To implement the tree, we'd instantiate multiple Axi4ReadOnlyArbiter and cascade them.
    // We fall back to a single one for simplicity in this generated code block.
    val arbiter = Axi4ReadOnlyArbiter(axiConfig, allAxiMasters.length)
    for (i <- allAxiMasters.indices) {
      arbiter.io.inputs(i) <> allAxiMasters(i)
    }
    io.axiMaster <> arbiter.io.output
  }
}
