package spinalML.examples

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.tensors.Tensor
import spinalML.layers.Conv2D

/**
 * DMATemplate: An Accelerator that autonomously fetches its image and weights
 * from external memory using an AXI Arbiter and DMA controllers.
 * 
 * Architecture:
 * - DMAReader2D fetches an 8x8 image patch by patch
 * - DMAReader (1D) fetches a 3x3 weight kernel
 * - DMAReader (1D) fetches the bias
 * - Axi4ReadOnlyArbiter merges their AXI requests to the DDR4 RAM
 * - Conv2D computes the convolution natively
 */
case class DMATemplate(dataType: HardType[Data]) extends Component {
  // AXI4 Config: 64-bit data bus (simulating a DDR4 connection)
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  // The Arbiter adds log2Up(3) = 2 bits to the ID for routing. So inputs must have idWidth = 2.
  val dmaAxiConfig = axiConfig.copy(idWidth = 2)

  val io = new Bundle {
    val start = slave(Event) // Standard stream event
    val imgAddr = in UInt(32 bits)
    val weightAddr = in UInt(32 bits)
    val biasAddr = in UInt(32 bits)
    
    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(dataType, Seq(36, 1), lanes = 1)) // 8x8 image, 3x3 kernel => 6x6 out = 36
  }

  // Fork the start event into 3 synchronized start triggers
  val startTriggers = StreamFork(io.start, 3)

  // ==========================================
  // 1. DMA for Image (8x8)
  // ==========================================
  val dmaImg = DMAReader2D(dataType, Seq(8, 8), outLanes = 1, dmaAxiConfig)
  dmaImg.io.cmd.valid := startTriggers(0).valid
  startTriggers(0).ready := dmaImg.io.cmd.ready
  dmaImg.io.cmd.baseAddress := io.imgAddr
  dmaImg.io.cmd.stride := 8 * (dataType.getBitsWidth / 8) // 8 pixels per row, converted to bytes
  dmaImg.io.cmd.patchWidth := (8 / (64 / dataType.getBitsWidth)) - 1
  dmaImg.io.cmd.patchHeight := 8

  // ==========================================
  // 2. DMA for Weights (3x3 = 9 elements)
  // ==========================================
  val dmaWeights = DMAReader(dataType, Seq(9, 1), outLanes = 9, dmaAxiConfig)
  val reqWeights = Stream(FetchRequest(32))
  reqWeights.valid := startTriggers(1).valid
  startTriggers(1).ready := reqWeights.ready
  reqWeights.address := io.weightAddr
  reqWeights.length := 2 // 3 beats total
  dmaWeights.io.cmd << reqWeights

  // ==========================================
  // 3. DMA for Bias (1 element)
  // ==========================================
  val dmaBias = DMAReader(dataType, Seq(1, 1), outLanes = 1, dmaAxiConfig)
  val reqBias = Stream(FetchRequest(32))
  reqBias.valid := startTriggers(2).valid
  startTriggers(2).ready := reqBias.ready
  reqBias.address := io.biasAddr
  reqBias.length := 0 // 1 beat
  dmaBias.io.cmd << reqBias

  // ==========================================
  // 4. AXI4 Arbiter (Memory Multiplexer)
  // ==========================================
  val arbiter = Axi4ReadOnlyArbiter(axiConfig, 3)
  arbiter.io.inputs(0) <> dmaImg.io.axiMaster
  arbiter.io.inputs(1) <> dmaWeights.io.axiMaster
  arbiter.io.inputs(2) <> dmaBias.io.axiMaster
  io.axiMaster <> arbiter.io.output

  // ==========================================
  // 5. Math Layer (Conv2D) with FIFOs to avoid AXI deadlock
  // ==========================================
  // If the AXI Arbiter sends Weights data but Conv2D is waiting for Image data,
  // it would block the entire AXI bus. We add queues to absorb the fetched data.
  val imgQueue = Tensor(dataType, Seq(8, 8), 1)
  imgQueue.stream << dmaImg.io.outStream.stream.queue(64)
  
  val weightsQueue = Tensor(dataType, Seq(9, 1), 9)
  weightsQueue.stream << dmaWeights.io.outStream.stream.queue(16)
  
  val biasQueue = Tensor(dataType, Seq(1, 1), 1)
  biasQueue.stream << dmaBias.io.outStream.stream.queue(16)

  val convOut = Conv2D(imgQueue, weightsQueue, biasQueue, dataType)
  io.outStream <> convOut
}
