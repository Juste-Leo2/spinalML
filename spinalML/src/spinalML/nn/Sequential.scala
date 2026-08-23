package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.tensors.Tensor
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

  // --- 1. Compute Shapes and Final Type ---
  def computeFinalShape(): (Seq[Int], HardType[Data]) = {
    var shape = inputShape
    var tpe = globalDataType
    for (layer <- layers) {
      shape = layer.getOutShape(shape)
      tpe = layer.outType(tpe)
    }
    (shape, tpe)
  }
  val (finalShape, finalType) = computeFinalShape()

  val io = new Bundle {
    val start = slave(Event)
    val imgBaseAddress = in UInt(axiConfig.addressWidth bits)
    val weightsBaseAddress = in UInt(axiConfig.addressWidth bits)
    
    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(finalType, finalShape, lanes = 1)) // Default to 1 lane output for now
  }

  // --- 2. Memory Offset Calculation & DMA Instantiation ---
  val allAxiMasters = scala.collection.mutable.ArrayBuffer[Axi4ReadOnly]()
  var currentMemoryOffset = 0
  
  // Start triggers fork
  // Image + each weight + each bias
  val totalDmaTriggers = 1 + layers.map(l => (if(l.getWeightShape().head > 0) 1 else 0) + (if(l.getBiasShape().head > 0) 1 else 0)).sum
  val startTriggers = StreamFork(io.start, totalDmaTriggers)
  var triggerIdx = 0
  
  // Base AXI config for leaf DMAs (accounting for arbiter routing bits)
  // For simplicity in this V1, we use a single stage arbiter if <= 16 ports.
  val dmaAxiConfig = axiConfig.copy(idWidth = axiConfig.idWidth - log2Up(Math.max(2, totalDmaTriggers)))

  // 2.1. Image DMA
  val inputDataType = globalDataType
  val dmaImg = DMAReader2D(inputDataType, inputShape, outLanes = 1, dmaAxiConfig)
  dmaImg.io.cmd.valid := startTriggers(triggerIdx).valid
  startTriggers(triggerIdx).ready := dmaImg.io.cmd.ready
  dmaImg.io.cmd.baseAddress := io.imgBaseAddress
  
  // Stride is typically width * byteSize
  val pixelBytes = inputDataType.getBitsWidth / 8
  dmaImg.io.cmd.stride := inputShape(1) * pixelBytes 
  val elementsPerBeat = axiConfig.dataWidth / inputDataType.getBitsWidth
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
  
  var currentTensor = imgQueue
  var currentShape = inputShape
  var currentType = globalDataType
  
  for (layer <- layers) {
    val lType = layer.outType(currentType)
    val wType = layer.weightType(currentType)
    
    val wShape = layer.getWeightShape()
    val bShape = layer.getBiasShape()
    
    var layerWeights: Tensor[Data] = null
    var layerBias: Tensor[Data] = null
    
    // Fetch Weights
    if (wShape.head > 0) {
      val elements = wShape.product
      val wLanes = if (layer.isInstanceOf[Conv2D]) wShape.head else wShape.head // Simplified lanes
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
      reqW.address := io.weightsBaseAddress + currentMemoryOffset
      val elementsPerBeat = axiConfig.dataWidth / wType.getBitsWidth
      val beats = (elements + elementsPerBeat - 1) / elementsPerBeat
      reqW.length := (if (beats > 0) beats - 1 else 0)
      dmaW.io.cmd << reqW
      
      allAxiMasters += dmaW.io.axiMaster
      triggerIdx += 1
      currentMemoryOffset += elements * (wType.getBitsWidth / 8)
      
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
      reqB.address := io.weightsBaseAddress + currentMemoryOffset
      
      val elementsPerBeatB = axiConfig.dataWidth / lType.getBitsWidth
      val beatsB = (elements + elementsPerBeatB - 1) / elementsPerBeatB
      reqB.length := (if (beatsB > 0) beatsB - 1 else 0)
      
      dmaB.io.cmd << reqB
      
      allAxiMasters += dmaB.io.axiMaster
      triggerIdx += 1
      currentMemoryOffset += elements * (lType.getBitsWidth / 8)
      
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
    val nextTensor = layer match {
      case c: Conv1D =>
        Conv1DHW(currentTensor, layerWeights, layerBias, lType)
        
      case c: Conv2D =>
        Conv2DHW(currentTensor, layerWeights, layerBias, lType)
        
      case _: ReLU =>
        relu(currentTensor)
        
      case lr: LeakyReLU =>
        leaky_relu(currentTensor, lr.shift)
        
      case _: Softmax =>
        val seqLen = currentTensor.shape(0)
        val channels = if (currentTensor.shape.length > 1) currentTensor.shape(1) else 1
        val comp = spinalML.activations.Softmax1D(currentType, channels, seqLen)
        comp.io.x <> currentTensor
        comp.io.y
        
      case bn: BatchNorm1D =>
        batchnorm(currentTensor, layerWeights, layerBias)
        
      case ln: LayerNorm1D =>
        val seqLen = currentTensor.shape(0)
        val channels = if (currentTensor.shape.length > 1) currentTensor.shape(1) else 1
        val comp = spinalML.layers.LayerNorm1D(currentType, channels, seqLen)
        comp.io.x <> currentTensor
        comp.io.gamma <> layerWeights
        comp.io.beta <> layerBias
        comp.io.y
        
      case mp: MaxPool1D =>
        val c = if (currentTensor.shape.length > 1) currentTensor.shape(1) else 1
        val repacked = if (currentTensor.lanes != c) repack(currentTensor, c) else currentTensor
        maxpool1d(repacked, mp.poolSize, mp.stride)
        
      case ap: AvgPool1D =>
        val c = if (currentTensor.shape.length > 1) currentTensor.shape(1) else 1
        val repacked = if (currentTensor.lanes != c) repack(currentTensor, c) else currentTensor
        avgpool1d(repacked, ap.poolSize, ap.stride)

      case mp2: MaxPool2D =>
        // Pooling 2D consumes one element per beat (lanes = 1) and emits C lanes per beat.
        // Repack on both sides to preserve the lanes = 1 invariant between layers.
        val in = if (currentTensor.lanes != 1) repack(currentTensor, 1) else currentTensor
        val pooled = maxpool2d(in, mp2.poolSize, mp2.stride)
        if (pooled.lanes != 1) repack(pooled, 1) else pooled

      case ap2: AvgPool2D =>
        val in = if (currentTensor.lanes != 1) repack(currentTensor, 1) else currentTensor
        val pooled = avgpool2d(in, ap2.poolSize, ap2.stride)
        if (pooled.lanes != 1) repack(pooled, 1) else pooled

      case _: Sigmoid =>
        sigmoid(currentTensor)

      case _: Tanh =>
        tanh(currentTensor)

      case _: Cast =>
        cast(currentTensor, lType)

      case _: Flatten =>
        flatten(currentTensor)
        
      case l: Linear =>
        val reshaped = reshape(currentTensor, Seq(1, l.inFeatures))
        val repackedTensor = repack(reshaped, l.inFeatures)
        // Weight-only quantization (wXaY): SInt weights (I4/I8) + compile-time scale(s)
        // are dequantized to the activation float dtype inside the layer.
        val linOut = layerWeights.dataType() match {
          case _: SInt =>
            spinalML.layers.Linear(repackedTensor, layerWeights.asInstanceOf[Tensor[SInt]], layerBias, lType, l.weightScales)
          case _ =>
            LinearHW(repackedTensor, layerWeights, layerBias, lType)
        }
        reshape(linOut, Seq(l.outFeatures, 1))
        
      case rq: Requantize =>
        spinalML.ops.requantize(currentTensor, rq.targetType, rq.shift)
        
      case rp: Repack =>
        repack(currentTensor, rp.newLanes)
        
      case a: ClassicalAttention =>
        val seqLen = currentTensor.shape(0)
        // wType carries the (possibly quantized) weight dtype declared via
        // customWeightType; scales drive the in-layer dequantization.
        val comp = ClassicalAttentionHW(currentType, wType, lType, seqLen, a.embedDim, a.numHeads, currentTensor.lanes, layerWeights.lanes, weightScales = a.weightScales)
        comp.io.x <> currentTensor
        
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
    
    // Update state for next layer
    currentTensor = nextTensor
    currentShape = layer.getOutShape(currentShape)
    currentType = lType
  }
  
  // Output assignment
  io.outStream <> currentTensor

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
