package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.tensors.Tensor
import spinalML.layers.{Conv1D => Conv1DHW, Conv2D => Conv2DHW, Linear => LinearHW, batchnorm}
import spinalML.ops.{reshape, repack, flatten}
import spinalML.activations.{relu, leaky_relu}
import spinalML.poolings.{maxpool1d, avgpool1d}

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

  val imgQueue = Tensor(inputDataType, inputShape, 1)
  imgQueue.stream << dmaImg.io.outStream.stream.queue(64)
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
      
      layerWeights = Tensor(wType, wShape, requiredLanes)
      layerWeights.stream << dmaW.io.outStream.stream.queue(16) // FIFO to prevent deadlocks
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
      
      layerBias = Tensor(lType, bShape, requiredBiasLanes)
      layerBias.stream << dmaB.io.outStream.stream.queue(16) // FIFO to prevent deadlocks
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
        
      case _: Flatten =>
        flatten(currentTensor)
        
      case l: Linear =>
        val reshaped = reshape(currentTensor, Seq(1, l.inFeatures))
        val repackedTensor = repack(reshaped, l.inFeatures)
        val linOut = LinearHW(repackedTensor, layerWeights, layerBias, lType)
        reshape(linOut, Seq(l.outFeatures, 1))
        
      case rq: Requantize =>
        spinalML.ops.requantize(currentTensor, rq.targetType, rq.shift)
        
      case rp: Repack =>
        repack(currentTensor, rp.newLanes)
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
