package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor

/**
 * FetchRequest2D specifies a 2D memory patch to fetch.
 * patchWidth: N-1 AXI beats per row (advisory; the component derives its own
 *             beat budget from stride/width so sub-beat row widths work)
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
 *
 * Rows are strictly serialized here (the next row command is issued only once
 * the previous row's output has fully crossed the trim stage), which makes row
 * boundaries exact regardless of internal pipelining in the 1D reader.
 *
 * Unaligned rows are supported: when a row's start address falls inside an AXI
 * beat (stride not a multiple of the bus width, e.g. 28-byte rows on a 64-bit
 * bus), the fetch is aligned DOWN to the beat boundary and the leading and
 * trailing extra elements are trimmed from the element stream, which then
 * carries exactly `shape(1)` elements per row.
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

  val readerCmd = Stream(FetchRequest(axiConfig.addressWidth))
  reader1D.io.cmd << readerCmd

  // --------------------------------------------------------
  // Geometry helpers
  // --------------------------------------------------------
  val elemBytes   = dataType.getBitsWidth / 8
  val bytesPerBeat = axiConfig.dataWidth / 8
  val elemsPerWord = bytesPerBeat / elemBytes
  val rowWidth     = U(shape(1), 16 bits)

  // --------------------------------------------------------
  // Address Generator State Machine
  // --------------------------------------------------------
  val currentAddress = Reg(UInt(axiConfig.addressWidth bits)) init (0)
  val currentRow     = Reg(UInt(8 bits)) init (0)
  val lastRow        = Reg(Bool()) init (False)
  val cmdHeight      = Reg(UInt(8 bits)) init (0)
  val cmdStride      = Reg(UInt(axiConfig.addressWidth bits)) init (0)

  // Per-row latched geometry (computed combinationally from currentAddress
  // while issuing the row command, then stable during drain)
  val rowReqAddr  = Reg(UInt(axiConfig.addressWidth bits)) init (0)
  val rowWords    = Reg(UInt(9 bits)) init (0)
  val rowSkip     = Reg(UInt(9 bits)) init (0) // leading elements to trim
  val rowKeepEnd  = Reg(UInt(12 bits)) init (0)// last kept element index

  readerCmd.valid   := False
  readerCmd.address := rowReqAddr
  readerCmd.length  := (rowWords -^ U(1, 9 bits)).resize(16 bits)

  io.cmd.ready := False

  // Combinational geometry for the row starting at currentAddress
  val headSkipBytes = currentAddress % U(bytesPerBeat, axiConfig.addressWidth bits)
  val headSkipElems = headSkipBytes >> log2Up(elemBytes)
  val reqAddrAligned = currentAddress - headSkipBytes
  val wordsForCurrentRow = ((headSkipElems +^ rowWidth +^ U(elemsPerWord - 1, 12 bits)) /
                            U(elemsPerWord, 12 bits)).resize(9 bits)

  // Output stream beats per fetched row
  val totalFetchElems = rowWords * U(elemsPerWord, 10 bits)
  val rowFetchedBeats = ((totalFetchElems / outLanes).resize(12 bits)) +
                        Mux(totalFetchElems % outLanes =/= 0, U(1, 12 bits), U(0, 12 bits))

  // --------------------------------------------------------
  // Row trim: mask leading (alignment) and trailing (overshoot) elements of
  // each row's raw stream so the output carries exactly `shape(1)` elements
  // per row. With aligned, exact-width rows this is a pure passthrough.
  // --------------------------------------------------------
  val elemCnt  = Reg(UInt(12 bits)) init (0)
  val suppress = (elemCnt < rowSkip.resize(13 bits)) || (elemCnt > rowKeepEnd)

  io.outStream.stream.valid := reader1D.io.outStream.stream.valid && !suppress
  reader1D.io.outStream.stream.ready := io.outStream.stream.ready || suppress
  io.outStream.stream.payload := reader1D.io.outStream.stream.payload

  when(reader1D.io.outStream.stream.fire) {
    elemCnt := Mux(elemCnt === rowFetchedBeats -^ U(1, 12 bits), U(0, 12 bits), elemCnt + 1)
  }

  val fsm = new StateMachine {
    val stateIdle: State = new State with EntryPoint {
      whenIsActive {
        when(io.cmd.valid) {
          currentAddress := io.cmd.baseAddress
          currentRow     := 0
          cmdHeight      := io.cmd.patchHeight
          cmdStride      := io.cmd.stride
          goto(stateFetch)
        }
      }
    }

    val stateFetch: State = new State {
      whenIsActive {
        readerCmd.valid := True
        readerCmd.address := reqAddrAligned
        readerCmd.length := (wordsForCurrentRow -^ U(1, 9 bits)).resize(16 bits)
        when(readerCmd.ready) {
          rowReqAddr := reqAddrAligned
          rowWords   := wordsForCurrentRow
          rowSkip    := headSkipElems.resize(9 bits)
          rowKeepEnd := (headSkipElems +^ rowWidth -^ U(1, 12 bits)).resize(12 bits)
          lastRow    := currentRow === cmdHeight - 1
          goto(stateDrain)
        }
      }
    }

    val stateDrain: State = new State {
      whenIsActive {
        when(reader1D.io.outStream.stream.fire && elemCnt === rowFetchedBeats -^ U(1, 12 bits)) {
          currentAddress := currentAddress + cmdStride
          currentRow     := currentRow + 1
          when(lastRow) {
            io.cmd.ready := True
            goto(stateIdle)
          } otherwise {
            goto(stateFetch)
          }
        }
      }
    }
  }



}
