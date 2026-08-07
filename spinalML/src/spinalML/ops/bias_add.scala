package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

/**
 * BiasAddOp: Broadcast Addition.
 * Takes a stream `A` of arbitrary shape, and a scalar stream `B`.
 * Loads the scalar `B` once, then adds it to all elements of `A` until the tensor `A` is fully processed.
 */
case class BiasAddOp[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val elements = shape.product
  require(elements % lanes == 0, "Total elements must be divisible by lanes")
  val cycles = elements / lanes

  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, lanes))
    val b = slave(Tensor(dataType, Seq(1, 1), 1)) // Bias is always a single scalar
    val c = master(Tensor(dataType, shape, lanes))
  }

  val biasReg = Reg(dataType)
  biasReg.init(biasReg.getZero.asInstanceOf[T])
  
  val counter = Counter(cycles)
  
  // Default values
  io.a.stream.ready := False
  io.b.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload.foreach(_.assignFromBits(B(0, widthOf(dataType) bits)))
  
  val fsm = new StateMachine {
    val stateLoadBias: State = new State with EntryPoint {
      whenIsActive {
        io.b.stream.ready := True
        when(io.b.stream.valid) {
          biasReg := io.b.stream.payload(0)
          goto(stateProcess)
        }
      }
    }
    
    val stateProcess: State = new State {
      whenIsActive {
        // Transparent passthrough for backpressure
        io.a.stream.ready := io.c.stream.ready
        io.c.stream.valid := io.a.stream.valid
        
        // Broadcast addition
        for (i <- 0 until lanes) {
          (io.a.stream.payload(i), biasReg) match {
            case (valA: SInt, valB: SInt) => io.c.stream.payload(i).assignFrom((valA + valB).asInstanceOf[T])
            case (valA: UInt, valB: UInt) => io.c.stream.payload(i).assignFrom((valA + valB).asInstanceOf[T])
            case _ => throw new Exception("Type unsupported")
          }
        }
        
        when(io.a.stream.fire) {
          counter.increment()
          when(counter.willOverflowIfInc) {
            goto(stateLoadBias)
          }
        }
      }
    }
  }
}

object bias_add {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(b.shape == Seq(1, 1), "Bias must be a scalar of shape [1, 1]")
    require(b.lanes == 1, "Bias must have 1 lane")
    
    val comp = BiasAddOp(a.dataType, a.shape, a.lanes)
    comp.io.a <> a
    comp.io.b <> b
    comp.io.c
  }
}
