// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinal.lib.bus.amba4.axi._
import spinalML.tensors.Tensor

/**
 * FetchRequest2D specifies a 2D memory patch to fetch.
 * baseAddress: Byte address of the first row's start
 * stride: Address offset (in bytes) between the start of row N and row N+1
 * patchHeight: Number of rows to fetch
 * patchWidth: IGNORED by the hardware (kept for API compatibility) — the
 *             component derives its own beat budget from stride/width so
 *             sub-beat row widths work.
 */
case class FetchRequest2D(addressWidth: Int) extends Bundle {
  val baseAddress = UInt(addressWidth bits)
  val stride      = UInt(addressWidth bits)
  val patchWidth  = UInt(16 bits)
  val patchHeight = UInt(16 bits)
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
 *
 * Preconditions:
 *  - Element dtype must be at least one byte wide (byte-addressed fetch).
 *  - Row width must be divisible by `outLanes`: the trim masks whole output
 *    beats and cannot split an element group across a row boundary.
 *  - When `outLanes > 1`, callers must additionally guarantee that every row
 *    start address falls on an output-group boundary (headSkipElems % outLanes
 *    == 0). Row addresses are runtime values, so this is checked by formal
 *    properties rather than at elaboration time (see roadmap §8).
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
  require(shape.length >= 2,
    "DMAReader2D: shape must be at least 2D (H, W) — it describes a patch of rows")
  require(dataType.getBitsWidth >= 8,
    s"DMAReader2D: ${dataType.getBitsWidth}-bit elements are unsupported — the fetch is byte-addressed, use a dtype of at least 8 bits")
  require(shape(1) % outLanes == 0,
    s"DMAReader2D: row width ${shape(1)} is not divisible by $outLanes output lanes — the beat-granular trim cannot split an element group across rows")
  val elemBytes   = dataType.getBitsWidth / 8
  val bytesPerBeat = axiConfig.dataWidth / 8
  val elemsPerWord = bytesPerBeat / elemBytes
  require(elemsPerWord >= 1,
    s"DMAReader2D: ${dataType.getBitsWidth}-bit elements are wider than the ${axiConfig.dataWidth}-bit AXI beat — unsupported element size")
  // Counter widths derived from the static shape (and worst-case row
  // misalignment) instead of fixed 8/9/12-bit ceilings, so any row width /
  // patch height is supported rather than silently wrapping.
  val maxRowWords    = (shape(1) + elemsPerWord - 1) / elemsPerWord + 1 // +1: alignment slack
  val rowWordsW      = log2Up(maxRowWords + 1)
  val skipW          = log2Up(elemsPerWord max 2)
  val keepEndW       = log2Up(shape(1) + elemsPerWord + 1)
  val maxFetchElems  = maxRowWords * elemsPerWord
  val beatsPerRowMax = (maxFetchElems + outLanes - 1) / outLanes
  val beatsW         = log2Up(beatsPerRowMax + 1)
  val rowWidth     = U(shape(1), log2Up(shape(1) + 1) bits)

  // --------------------------------------------------------
  // Address Generator State Machine
  // --------------------------------------------------------
  val currentAddress = Reg(UInt(axiConfig.addressWidth bits)) init (0)
  val currentRow     = Reg(UInt(16 bits)) init (0)
  val lastRow        = Reg(Bool()) init (False)
  val cmdHeight      = Reg(UInt(16 bits)) init (0)
  val cmdStride      = Reg(UInt(axiConfig.addressWidth bits)) init (0)

  // Per-row latched geometry (computed combinationally from currentAddress
  // while issuing the row command, then stable during drain)
  val rowReqAddr  = Reg(UInt(axiConfig.addressWidth bits)) init (0)
  val rowWords    = Reg(UInt(rowWordsW bits)) init (0)
  val rowSkip     = Reg(UInt(skipW bits)) init (0) // leading elements to trim
  val rowKeepEnd  = Reg(UInt(keepEndW bits)) init (0)// last kept element index

  readerCmd.valid   := False
  readerCmd.address := rowReqAddr
  readerCmd.length  := (rowWords -^ U(1, rowWordsW bits)).resize(16 bits)

  io.cmd.ready := False

  // Combinational geometry for the row starting at currentAddress
  val headSkipBytes = currentAddress % U(bytesPerBeat, axiConfig.addressWidth bits)
  val headSkipElems = headSkipBytes >> log2Up(elemBytes)
  val reqAddrAligned = currentAddress - headSkipBytes
  val wordsForCurrentRow = ((headSkipElems +^ rowWidth +^ U(elemsPerWord - 1)) /
                            U(elemsPerWord)).resize(rowWordsW bits)

  // Output stream beats per fetched row
  val totalFetchElems = rowWords * U(elemsPerWord)
  val rowFetchedBeats = ((totalFetchElems / outLanes).resize(beatsW bits)) +
                        Mux(totalFetchElems % outLanes =/= 0, U(1, beatsW bits), U(0, beatsW bits))

  // --------------------------------------------------------
  // Row trim: mask leading (alignment) and trailing (overshoot) elements of
  // each row's raw stream so the output carries exactly `shape(1)` elements
  // per row. With aligned, exact-width rows this is a pure passthrough.
  // --------------------------------------------------------
  val elemCnt  = Reg(UInt(beatsW bits)) init (0)
  val suppress = (elemCnt < rowSkip.resize(beatsW bits)) || (elemCnt > rowKeepEnd)

  io.outStream.stream.valid := reader1D.io.outStream.stream.valid && !suppress
  reader1D.io.outStream.stream.ready := io.outStream.stream.ready || suppress
  io.outStream.stream.payload := reader1D.io.outStream.stream.payload

  when(reader1D.io.outStream.stream.fire) {
    elemCnt := Mux(elemCnt === rowFetchedBeats -^ U(1, beatsW bits), U(0, beatsW bits), elemCnt + 1)
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
        readerCmd.length := (wordsForCurrentRow -^ U(1, rowWordsW bits)).resize(16 bits)
        when(readerCmd.ready) {
          rowReqAddr := reqAddrAligned
          rowWords   := wordsForCurrentRow
          rowSkip    := headSkipElems.resize(skipW bits)
          rowKeepEnd := (headSkipElems +^ rowWidth -^ U(1)).resize(keepEndW bits)
          lastRow    := currentRow === cmdHeight - 1
          goto(stateDrain)
        }
      }
    }

    val stateDrain: State = new State {
      whenIsActive {
        when(reader1D.io.outStream.stream.fire && elemCnt === rowFetchedBeats -^ U(1, beatsW bits)) {
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
