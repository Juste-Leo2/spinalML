package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

case class ConcatenateAxis0Op[T <: Data](dataType: HardType[T], shapeA: Seq[Int], shapeB: Seq[Int], lanes: Int) extends Component {
  require(shapeA.tail == shapeB.tail, "Tensors must have the same shape except for the concatenation axis (axis 0)")
  
  val L_A = shapeA.head
  val L_B = shapeB.head
  val L_out = L_A + L_B
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, lanes))
    val c = master(Tensor(dataType, Seq(L_out) ++ shapeA.tail, lanes))
  }
  
  val countA = Counter(L_A)
  val countB = Counter(L_B)
  
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
