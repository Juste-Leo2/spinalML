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
case class BiasAddOp[T <: Data](dataType: HardType[T], shapeA: Seq[Int], shapeB: Seq[Int], lanes: Int) extends Component {
  val M = if (shapeA.length > 1) shapeA(0) else 1
  val N = shapeA.last
  require(shapeB.last == N, "Bias vector length must match the last dimension of A")
  
  val elements = shapeA.product
  require(elements % lanes == 0, "Total elements of A must be divisible by lanes")
  val cycles = elements / lanes

  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, 1)) // Bias arrives sequentially with lanes=1
    val c = master(Tensor(dataType, shapeA, lanes))
    val reArm = in Bool() // command-boundary re-arm (abort pass, re-load bias)
  }

  // Memory to store the N bias elements
  // We use Vec(Reg) for async read since N is typically a feature dimension
  // and we might need to read multiple elements per cycle if lanes > 1.
  val biasMem = Vec(Reg(dataType), N)
  biasMem.foreach(_.init(biasMem.head.getZero.asInstanceOf[T]))
  
  val loadCounter = Counter(N)
  val aCounter = Counter(cycles)
  
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
          biasMem(loadCounter.value) := io.b.stream.payload(0)
          loadCounter.increment()
          when(loadCounter.willOverflowIfInc) {
            goto(stateProcess)
          }
        }
        when(io.reArm) {
          loadCounter.clear()
        }
      }
    }
    
    val stateProcess: State = new State {
      whenIsActive {
        // Transparent passthrough for backpressure
        io.a.stream.ready := io.c.stream.ready
        io.c.stream.valid := io.a.stream.valid
        
        // Calculate the starting column index for this chunk
        val startCol = (aCounter.value * lanes) % N
        
        // Broadcast addition
        for (i <- 0 until lanes) {
          val colIdx = (startCol + i) % N
          val biasVal = biasMem(colIdx.resized)
          
          (io.a.stream.payload(i), biasVal) match {
            case (valA: SInt, valB: SInt) => io.c.stream.payload(i).assignFrom((valA + valB).asInstanceOf[T])
            case (valA: UInt, valB: UInt) => io.c.stream.payload(i).assignFrom((valA + valB).asInstanceOf[T])
            case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => io.c.stream.payload(i).assignFrom(spinalML.utils.Float.add(valA, valB).asInstanceOf[T])
            case _ => throw new Exception("Type unsupported")
          }
        }
        
        when(io.a.stream.fire) {
          aCounter.increment()
          when(aCounter.willOverflowIfInc) {
            goto(stateDone)
          }
        }

        // Command-boundary re-arm: abandon the current pass and wait for a
        // fresh bias load (prevents stale generation-bias on the last tile).
        when(io.reArm) {
          aCounter.clear()
          goto(stateLoadBias)
        }
      }
    }
    
    val stateDone: State = new State {
       whenIsActive {
         aCounter.clear()
         loadCounter.clear()
         goto(stateLoadBias)
       }
    }
  }
}

object bias_add {
  def apply[T <: Data](a: Tensor[T], b: Tensor[T], reArm: Option[Bool] = None): Tensor[T] = {
    require(b.lanes == 1, "Bias must have 1 lane for BiasAddOp")
    
    val comp = BiasAddOp(a.dataType, a.shape, b.shape, a.lanes)
    comp.io.a <> a
    comp.io.b <> b
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.c
  }
}
