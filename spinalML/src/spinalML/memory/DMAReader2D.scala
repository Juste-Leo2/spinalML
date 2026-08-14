package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor

/**
 * FetchRequest2D specifies a 2D memory patch to fetch.
 * patchWidth: N-1 AXI beats per row
 * patchHeight: Number of rows to fetch
 * stride: Address offset (in bytes) between the start of row N and row N+1
 */
case class FetchRequest2D(addressWidth: Int) extends Bundle {
  val baseAddress = UInt(addressWidth bits)
  val stride      = UInt(addressWidth bits)
  val patchWidth  = UInt(8 bits)
  val patchHeight = UInt(8 bits)
}

/**
 * DMAReader2D: An AXI4-Master module that fetches 2D data patches (e.g., images) from DDR4.
 * It acts as an Address Generator, breaking a 2D request into multiple 1D row requests
 * and feeding them to an internal 1D DMAReader for extreme code reuse and optimization.
 */
case class DMAReader2D[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  outLanes: Int,
  axiConfig: Axi4Config
) extends Component {

  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest2D(axiConfig.addressWidth)))
    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(dataType, shape, outLanes))
  }

  // Instantiation of the highly optimized 1D reader
  val reader1D = DMAReader(dataType, shape, outLanes, axiConfig)
  io.axiMaster <> reader1D.io.axiMaster
  io.outStream <> reader1D.io.outStream

  val readerCmd = Stream(FetchRequest(axiConfig.addressWidth))
  reader1D.io.cmd << readerCmd

  // --------------------------------------------------------
  // Address Generator State Machine
  // --------------------------------------------------------
  val currentAddress = Reg(UInt(axiConfig.addressWidth bits))
  val currentRow     = Reg(UInt(8 bits))
  
  readerCmd.valid   := False
  readerCmd.address := currentAddress
  readerCmd.length  := io.cmd.patchWidth
  
  io.cmd.ready := False
  
  val fsm = new StateMachine {
    val stateIdle: State = new State with EntryPoint {
      whenIsActive {
        when(io.cmd.valid) {
          currentAddress := io.cmd.baseAddress
          currentRow     := 0
          goto(stateFetch)
        }
      }
    }
    
    val stateFetch: State = new State {
      whenIsActive {
        readerCmd.valid := True
        when(readerCmd.ready) {
          currentAddress := currentAddress + io.cmd.stride
          currentRow     := currentRow + 1
          
          when(currentRow === io.cmd.patchHeight - 1) {
            io.cmd.ready := True
            goto(stateIdle)
          }
        }
      }
    }
  }
}
