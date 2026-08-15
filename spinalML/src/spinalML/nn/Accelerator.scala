package spinalML.nn

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite._
import spinalML.tensors.Tensor

/**
 * Top-level wrapper that converts a generic `Sequential` model into a complete
 * System-on-Chip (SoC) ready hardware accelerator.
 * 
 * It automatically exposes two standard memory-mapped buses:
 * - AXI4 Master (High Speed): For fetching features and weights directly from DDR.
 * - AXI4-Lite Slave (Control): For the CPU to configure registers and start the inference.
 */
class Accelerator[T <: Data](
  val dataType: HardType[T],
  val inputShape: Seq[Int],
  val modelSpec: Seq[spinalML.nn.LayerSpec],
  val axiConfig: Axi4Config
) extends Component {

  val axiLiteConfig = AxiLite4Config(addressWidth = 8, dataWidth = 32)
  
  // Cast interne invisible pour l'utilisateur
  val globalDataType = dataType.asInstanceOf[HardType[Data]]
  
  // 1. Instantiate the neural network datapath first to infer its output shape
  val model = Sequential(globalDataType, inputShape, modelSpec, axiConfig)
  
  val io = new Bundle {
    // High-speed Master for DDR access
    val axiMaster = master(Axi4(axiConfig))
    
    // Low-speed Slave for CPU configuration
    val ctrlBus = slave(AxiLite4(axiLiteConfig))
    
    // Output Stream for the final result
    val outStream = master(cloneOf(model.io.outStream))
  }
  
  // 2. Map the AXI4 Master
  // We connect the Read channels. Write channels are grounded since we only infer (read-only DDR).
  io.axiMaster.ar << model.io.axiMaster.ar
  model.io.axiMaster.r << io.axiMaster.r
  
  io.axiMaster.aw.valid := False
  io.axiMaster.aw.payload.assignDontCare()
  io.axiMaster.w.valid := False
  io.axiMaster.w.payload.assignDontCare()
  io.axiMaster.b.ready := False
  
  // 3. Map the final output stream
  io.outStream <> model.io.outStream
  
  // 4. Create the AXI4-Lite Control Registers
  val ctrlFactory = new AxiLite4SlaveFactory(io.ctrlBus)
  
  // Register 0x00: Control
  // Bit 0: Start inference (trigger)
  val startPending = RegInit(False)
  ctrlFactory.onWrite(0x00) {
    // Trigger inference. We hold the request until the datapath accepts it.
    startPending := True
  }
  
  val startEvent = Event
  startEvent.valid := startPending
  when(startEvent.fire) {
    startPending := False
  }
  
  model.io.start << startEvent
  
  // Register 0x04: Status
  // Bit 0: Done (We can read it if we want, currently tied to outStream valid)
  ctrlFactory.read(io.outStream.stream.valid, 0x04, 0)
  
  // Register 0x08: Image Base Address
  val imgAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), 0x08, 0) init(0)
  model.io.imgBaseAddress := imgAddrReg
  
  // Register 0x0C: Weights Base Address
  val weightsAddrReg = ctrlFactory.createReadAndWrite(UInt(axiConfig.addressWidth bits), 0x0C, 0) init(0)
  model.io.weightsBaseAddress := weightsAddrReg
}
