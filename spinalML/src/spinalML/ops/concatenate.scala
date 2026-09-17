// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

case class ConcatenateAxis0Op[T <: Data](dataType: HardType[T], shapeA: Seq[Int], shapeB: Seq[Int], lanes: Int) extends Component {
  require(shapeA.tail == shapeB.tail, "Tensors must have the same shape except for the concatenation axis (axis 0)")

  // OPS-04 contract: the FSM counts streamed beats, not axis-0 cells.
  // Row-major, one axis-0 cell = tailProduct elements = beatsPerRow beats.
  // 1D legacy: shape.head counts beats directly (all existing benches drive
  // `head` beats of `lanes` elements). Rows must be beat-aligned
  // (tailProduct % lanes == 0): a beat straddling two rows belongs to both
  // and is unrecoverable — same fail-fast rationale as OPS-07.
  private def beatsPerRow(shape: Seq[Int]): Int =
    if (shape.tail.isEmpty) 1
    else {
      val tailProduct = shape.tail.product
      require(tailProduct % lanes == 0,
        s"ConcatenateAxis0Op: tail shape ${shape.tail} ($tailProduct elements) is not a multiple of lanes=$lanes — rows would straddle beats (OPS-04: repack to a divisor of the tail product first)")
      tailProduct / lanes
    }

  val L_A = shapeA.head
  val L_B = shapeB.head
  val L_out = L_A + L_B
  val beatsA = L_A * beatsPerRow(shapeA)
  val beatsB = L_B * beatsPerRow(shapeB)

  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, lanes))
    val c = master(Tensor(dataType, Seq(L_out) ++ shapeA.tail, lanes))
  }

  val countA = Counter(beatsA)
  val countB = Counter(beatsB)
  
  io.a.stream.ready := False
  io.b.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload := io.a.stream.payload // Default to avoid latch
  
  val fsm = new StateMachine {
    val stateA: State = new State with EntryPoint {
      whenIsActive {
        io.c.stream.valid := io.a.stream.valid
        io.c.stream.payload := io.a.stream.payload
        io.a.stream.ready := io.c.stream.ready
        
        when(io.a.stream.valid && io.c.stream.ready) {
          countA.increment()
          when(countA.willOverflowIfInc) {
            goto(stateB)
          }
        }
      }
    }
    
    val stateB: State = new State {
      whenIsActive {
        io.c.stream.valid := io.b.stream.valid
        io.c.stream.payload := io.b.stream.payload
        io.b.stream.ready := io.c.stream.ready
        
        when(io.b.stream.valid && io.c.stream.ready) {
          countB.increment()
          when(countB.willOverflowIfInc) {
            goto(stateDone)
          }
        }
      }
    }
    
    val stateDone: State = new State {
      whenIsActive {
        countA.clear()
        countB.clear()
        goto(stateA)
      }
    }
  }
}

case class ConcatenateAxis1Op[T <: Data](dataType: HardType[T], shape: Seq[Int], lanesA: Int, lanesB: Int) extends Component {
  // OPS-05: the join pairs input beats lockstep, one output beat per input
  // beat pair. Correct if and only if each beat holds exactly one full row
  // on both sides (enforced by `concatenate.apply` below): sub-row lanes
  // would emit interleaved half-joins, unequal lanes deadlock StreamJoin.
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanesA))
    val b = slave(Tensor(dataType, shape, lanesB))
    val c = master(Tensor(dataType, shape, lanesA + lanesB))
  }
  
  val syncStream = StreamJoin.arg(io.a.stream, io.b.stream)
  io.c.stream.arbitrationFrom(syncStream)
  
  for(i <- 0 until lanesA) {
    io.c.stream.payload(i) := io.a.stream.payload(i)
  }
  for(i <- 0 until lanesB) {
    io.c.stream.payload(lanesA + i) := io.b.stream.payload(i)
  }
}

object concatenate {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T], axis: Int): Tensor[T] = {
    if (axis == 0) {
      require(a.lanes == b.lanes, "Tensors must have the same lanes for axis 0 concatenation")
      val comp = ConcatenateAxis0Op(a.dataType, a.shape, b.shape, a.lanes)
      comp.io.a <> a
      comp.io.b <> b
      comp.io.c
    } else if (axis == 1) {
      require(a.shape == b.shape, "Tensors must have same temporal shape for axis 1 concatenation")
      // OPS-05 contract: the per-beat lane join is a row-wise concat only
      // when each beat carries exactly one full row on both sides. Anything
      // else is silently wrong (sub-row lanes interleave half-rows) or hangs
      // (unequal lanes => unequal beat counts => StreamJoin starves).
      require(a.shape.length > 1,
        s"Concatenate axis=1 needs a row dimension (2D+ tensors), got shape ${a.shape}")
      require(a.lanes == b.lanes,
        s"Concatenate axis=1 needs equal lanes (lockstep beat pairing): lanesA=${a.lanes} != lanesB=${b.lanes}")
      require(a.lanes == a.shape(1),
        s"Concatenate axis=1 needs lanes == row width ${a.shape(1)} (one full row per beat): got lanes=${a.lanes} (repack first)")
      val newShape = if (a.shape.length > 1) {
        a.shape.updated(1, a.shape(1) + b.shape(1))
      } else {
        a.shape
      }
      val comp = ConcatenateAxis1Op(a.dataType, newShape, a.lanes, b.lanes)
      comp.io.a <> a
      comp.io.b <> b
      comp.io.c
    } else {
      throw new Exception(s"Concatenation along axis $axis not supported yet")
    }
  }
}
