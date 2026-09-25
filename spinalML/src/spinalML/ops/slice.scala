// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

case class SliceAxis0Op[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int, start: Int, end: Int) extends Component {
  val L_in = shape.head
  val L_out = end - start
  require(start >= 0 && end <= L_in && start < end, s"Invalid slice range [$start:$end] for length $L_in")

  // Beat-counting contract: start/end are axis-0 cell indices; one cell =
  // beatsPerRow beats on the wire (same beat-counting as ConcatenateAxis0Op).
  // 1D legacy: shape.head counts beats directly. Rows must be beat-aligned.
  val beatsPerRow: Int =
    if (shape.tail.isEmpty) 1
    else {
      val tailProduct = shape.tail.product
      require(tailProduct % lanes == 0,
        s"SliceAxis0Op: tail shape ${shape.tail} ($tailProduct elements) is not a multiple of lanes=$lanes — rows would straddle beats (OPS-04: repack to a divisor of the tail product first)")
      tailProduct / lanes
    }
  val totalBeats = L_in * beatsPerRow
  val keepStart = start * beatsPerRow
  val keepEnd = end * beatsPerRow // may equal totalBeats

  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, Seq(L_out) ++ shape.tail, lanes))
  }

  val counter = Counter(totalBeats)
  // Threshold widths match the counter width (log2Up(totalBeats)), exactly
  // like the legacy code: keepEnd == totalBeats is never elaborated as a
  // literal (first branch), so it always fits.
  val startU = U(keepStart, log2Up(totalBeats) bits)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload := io.a.stream.payload
  
  // `end` may equal L_in: everything from start onward is then kept.
  val inRange = if (keepEnd >= totalBeats) {
    counter.value >= startU
  } else {
    counter.value >= startU && counter.value < U(keepEnd, log2Up(totalBeats) bits)
  }
  
  when(counter.value < startU) {
    io.a.stream.ready := True
    when(io.a.stream.valid) {
      counter.increment()
    }
  } elsewhen(inRange) {
    io.c.stream.valid := io.a.stream.valid
    io.a.stream.ready := io.c.stream.ready
    when(io.a.stream.valid && io.c.stream.ready) {
      counter.increment()
    }
  } otherwise {
    io.a.stream.ready := True
    when(io.a.stream.valid) {
      counter.increment()
    }
  }
}

case class SliceAxis1Op[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int, start: Int, end: Int) extends Component {
  val L_out = end - start
  require(start >= 0 && end <= lanes && start < end, s"Invalid slice range [$start:$end] for lanes $lanes")
  
  val newShape = if (shape.length > 1) {
    shape.updated(1, L_out)
  } else {
    shape
  }
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, newShape, L_out)) // Lanes also reduced
  }
  
  io.c.stream.valid := io.a.stream.valid
  io.a.stream.ready := io.c.stream.ready
  for(i <- 0 until L_out) {
    io.c.stream.payload(i) := io.a.stream.payload(start + i)
  }
}

object slice {
  def apply[T <: Data](a: Tensor[T], start: Int, end: Int, axis: Int): Tensor[T] = {
    if (axis == 0) {
      val comp = SliceAxis0Op(a.dataType, a.shape, a.lanes, start, end)
      comp.io.a <> a
      comp.io.c
    } else if (axis == 1) {
      val comp = SliceAxis1Op(a.dataType, a.shape, a.lanes, start, end)
      comp.io.a <> a
      comp.io.c
    } else {
      throw new Exception(s"Slice along axis $axis not supported yet")
    }
  }
}
