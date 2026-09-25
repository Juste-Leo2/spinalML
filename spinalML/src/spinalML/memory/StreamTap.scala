// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

/**
 * StreamTap: on-chip A-operand replayer for a spilling layer whose input
 * is NOT DDR-backed (deep node, or a node-0 shared with another consumer — re-firing DDR there would push duplicate
 * beats into the other branch).
 *
 * Pass 0 (`arm` at the command boundary, then snoop mode): fully
 * transparent — `streamOut << streamIn`, zero added backpressure — while one
 * copy of every beat lands in the Mem.
 *
 * Passes > 0 (`replay` pulse per pass): `streamOut` re-emits the recorded
 * beats VERBATIM (same framing, padding included — bit-exact by
 * construction, no divisibility hypothesis), `streamIn.ready` held low so
 * unexpected upstream traffic backpressures loudly instead of aliasing.
 *
 * Sizing: `maxBeats = ceil(elems/lanes) + 1` (one TapBuffer-style slack
 * beat); a snoop overflow fails the sim loudly instead of wrapping
 * silently. Replay storage is a combinational-read Mem (LUTRAM): no skid
 * logic, affordable inside the S0 replay budget.
 */
case class StreamTap[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int
) extends Component {
  require(lanes >= 1, s"StreamTap lanes=$lanes must be >= 1")
  val elems = shape.product
  require(elems >= 1, s"StreamTap shape $shape must be non-empty")
  val elemBits = dataType.getBitsWidth
  require(elemBits >= 1, s"StreamTap dtype must have a positive width")
  val beatBits = lanes * elemBits
  val maxBeats = (elems + lanes - 1) / lanes + 1

  val mem = Mem(Bits(beatBits bits), maxBeats)

  val io = new Bundle {
    val streamIn = slave(Tensor(dataType, shape, lanes))
    val streamOut = master(Tensor(dataType, shape, lanes))
    val arm = in Bool() // command boundary: snoop mode, write pointer reset
    val replay = in Bool() // per-pass pulse (p > 0): replay mode, read pointer reset
  }

  val snooping = RegInit(True) // reset = snoop (pass 0 records)
  val wptr = Reg(UInt(log2Up(maxBeats) bits)) init(0)
  val rptr = Reg(UInt(log2Up(maxBeats) bits)) init(0)
  val count = Reg(UInt(log2Up(maxBeats) bits)) init(0) // beats recorded

  when(io.arm) {
    snooping := True
    wptr := 0
  }
  when(io.replay) {
    snooping := False
    rptr := 0
    count := wptr
  }

  // Replay carrier: combinational-read Mem, no skid. Defined before the
  // snoop mux below (Scala execution order).
  val rdata = mem.readAsync(rptr)
  val unpackedReplay = Tensor(dataType, shape, lanes)
  for (i <- 0 until lanes) {
    unpackedReplay.stream.payload(i).assignFromBits(rdata(i * elemBits, elemBits bits))
  }
  // `unpackedReplay` is payload-only (its stream handshake is unused — the
  // muxed streamOut handshake below governs); tie it idle so no
  // undriven-net check can fire on it.
  unpackedReplay.stream.valid := False

  // Snoop path: transparent passthrough + verbatim record (payload muxed
  // per lane — never Mux the whole Tensor, that would also mux valid/ready
  // against the explicit assignments below).
  val snoopFire = snooping && io.streamIn.stream.fire
  io.streamOut.stream.valid := Mux(snooping, io.streamIn.stream.valid, rptr < count)
  for (i <- 0 until lanes) {
    io.streamOut.stream.payload(i) := Mux(snooping,
      io.streamIn.stream.payload(i), unpackedReplay.stream.payload(i))
  }
  io.streamIn.stream.ready := Mux(snooping, io.streamOut.stream.ready, False)

  // Beat packing (lane order mirrored on unpack — verbatim round-trip).
  val wdata = Bits(beatBits bits)
  for (i <- 0 until lanes) {
    wdata(i * elemBits, elemBits bits) := io.streamIn.stream.payload(i).asBits
  }
  when(snoopFire) {
    assert(wptr < U(maxBeats), "StreamTap snoop overflow: more beats than ceil(elems/lanes)+1")
    mem.write(wptr, wdata)
    wptr := wptr + 1
  }

  when(!snooping && io.streamOut.stream.fire) {
    rptr := rptr + 1
  }
}
