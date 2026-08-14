package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor
import spinalML.ops.repack

/**
 * FetchRequest specifies where to read and how much.
 * length: Number of AXI beats minus 1 (AXI4 arlen format). 0 means 1 beat, 255 means 256 beats.
 */
case class FetchRequest(addressWidth: Int) extends Bundle {
  val address = UInt(addressWidth bits)
  val length  = UInt(8 bits)
}

/**
 * DMAReader: An AXI4-Master module that fetches data from DDR4 
 * and automatically adapts the physical bus width to the requested ML Tensor lanes.
 */
case class DMAReader[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  outLanes: Int,
  axiConfig: Axi4Config
) extends Component {

  val axiLanes = axiConfig.dataWidth / dataType.getBitsWidth
  require(axiConfig.dataWidth % dataType.getBitsWidth == 0, "AXI data width must be a multiple of the data type width")

  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest(axiConfig.addressWidth)))
    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(dataType, shape, outLanes))
  }

  // ==========================================
  // 1. AR (Address Read) Channel Handshake
  // ==========================================
  io.axiMaster.ar.valid := io.cmd.valid
  io.cmd.ready          := io.axiMaster.ar.ready
  
  io.axiMaster.ar.addr  := io.cmd.address
  io.axiMaster.ar.len   := io.cmd.length
  io.axiMaster.ar.size  := log2Up(axiConfig.dataWidth / 8) // Bytes per beat
  io.axiMaster.ar.burst := B"01" // INCR burst mode (linear memory fetch)
  
  // Default tied-off AXI4 signals
  io.axiMaster.ar.id    := 0
  io.axiMaster.ar.prot  := 0
  io.axiMaster.ar.cache := 0
  io.axiMaster.ar.lock  := 0
  io.axiMaster.ar.qos   := 0
  io.axiMaster.ar.region := 0

  // ==========================================
  // 2. R (Read Data) Channel Handshake -> Raw Tensor
  // ==========================================
  val axiRawTensor = Tensor(dataType, shape, axiLanes)
  
  axiRawTensor.stream.valid := io.axiMaster.r.valid
  io.axiMaster.r.ready      := axiRawTensor.stream.ready
  
  // Convert physical AXI bits into ML DataType array
  for (i <- 0 until axiLanes) {
    val slice = io.axiMaster.r.data(i * dataType.getBitsWidth, dataType.getBitsWidth bits)
    axiRawTensor.stream.payload(i).assignFromBits(slice)
  }

  // ==========================================
  // 3. Internal Gearbox (Repack)
  // ==========================================
  // Automatically adapt the raw physical AXI stream to the desired ML lanes.
  val repackedTensor = repack(axiRawTensor, outLanes)
  
  io.outStream <> repackedTensor
}
