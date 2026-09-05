// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

// Hardware Gearbox component to convert stream widths
case class RepackOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int, newLanes: Int,
                               withFlush: Boolean = false) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val c = master(Tensor(dataType, shape, newLanes))
    // Command-boundary flush (flushable gearbox only)
    val reArm = in Bool()
    // High when nothing is parked here (flushable gearbox only)
    val isEmpty = out Bool()
  }

  if (!withFlush) {
    // ---- Legacy SpinalHDL width adapter: battle-tested pacing everywhere
    //      the stream is guaranteed group-aligned (e.g. image rows). ----
    val bitStreamIn = io.a.stream.translateWith(io.a.stream.payload.asBits)
    val bitStreamOut = Stream(Bits(newLanes * widthOf(dataType) bits))
    StreamWidthAdapter(bitStreamIn, bitStreamOut)
    io.c.stream.arbitrationFrom(bitStreamOut)
    io.c.stream.payload.assignFromBits(bitStreamOut.payload)
    // io.reArm is intentionally ignored here: nothing is parked between
    // commands in this mode. The caller still ties it (see repack.apply).
    io.isEmpty := True

  } else {
    // ---- Flushable structured gearbox ----
    val inW = lanes * widthOf(dataType)
    val outW = newLanes * widthOf(dataType)

    if (lanes > newLanes) {
      // SPLIT mode: k sub-groups emitted per input beat (element 0 first)
      val k = lanes / newLanes
      val hold = Reg(Bits(inW bits))
      val idx = Reg(UInt(log2Up(k) bits)) init (0)
      val full = RegInit(False)

      io.a.stream.ready := !full || (io.c.stream.fire && idx === U(k - 1, log2Up(k) bits))
      io.c.stream.valid := full
      io.c.stream.payload.assignFromBits(hold(idx * outW, outW bits))

      // Reload wins over drain-end on a simultaneous last-beat handshake
      when(io.a.stream.fire) {
        hold := io.a.stream.payload.asBits
        idx := 0
        full := True
      } otherwise {
        when(io.c.stream.fire) {
          idx := idx + 1
          when(idx === U(k - 1, log2Up(k) bits)) { full := False }
        }
      }
      when(io.reArm) { full := False; idx := 0 }
      io.isEmpty := !full

    } else {
      // AGGREGATE mode: m input beats collected per output group
      val m = newLanes / lanes
      val collect = Reg(Bits(outW bits))
      val idx = Reg(UInt(log2Up(m) bits)) init (0)
      val full = RegInit(False)

      io.a.stream.ready := !full
      io.c.stream.valid := full
      io.c.stream.payload.assignFromBits(collect)

      when(io.a.stream.fire) {
        collect(idx * inW, inW bits) := io.a.stream.payload.asBits
        when(idx === U(m - 1, log2Up(m) bits)) { full := True }
        idx := idx + 1
      }
      when(io.c.stream.fire) { full := False; idx := 0 }
      when(io.reArm) { full := False; idx := 0 }
      io.isEmpty := !full
    }
  }
}

object repack {
  /**
   * Modifies the physical lane width of a Tensor stream (Gearbox).
   * For instance, converts a 64-lane tensor stream into a 32-lane stream.
   * This does not modify the logical ML shape of the tensor.
   *
   * withFlush = true (structured gearbox) attaches a hard `ready := !full`
   * to the caller's upstream. ATTACHMENT RULE (bisection M1.7,
   * docs/open-mysteries.md): never hang this directly onto a shared fan-out
   * branch without a local elastic stage — use a depth>=2 FIFO or a pipe pair
   * at the attach point. Safe usages today: inside DMAReaders behind the
   * empty-gated cmd.ready (weight/bias path). Known-unsafe: the Linear-input
   * repack in nn/Sequential.scala (see guard comment there).
   */
  def apply[T <: Data](a: Tensor[T], newLanes: Int, reArm: Option[Bool] = None,
                       created: scala.collection.mutable.ArrayBuffer[RepackOp[_]] = null,
                       withFlush: Boolean = false): Tensor[T] = {
    // If the lanes are already correct, return the tensor directly
    if (a.lanes == newLanes) return a

    def mk(old: Tensor[T], nl: Int): Tensor[T] = {
      val repackComp = RepackOp(old.dataType, old.shape, old.lanes, nl, withFlush)
      repackComp.io.reArm := reArm.getOrElse(False)
      if (created != null) created += repackComp
      repackComp.io.a <> old
      repackComp.io.c
    }

    // If widths are not multiples of each other, chain through 1 lane
    if (a.lanes % newLanes != 0 && newLanes % a.lanes != 0) {
      val temp = mk(a, 1)
      return mk(temp, newLanes)
    }

    mk(a, newLanes)
  }
}
