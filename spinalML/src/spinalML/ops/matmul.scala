package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

/**
 * MatmulOp: Matrix-Vector multiplication using SRAM (Weight Stationary) + MAC architecture.
 * A is [M, K], B is [K, 1] (Vector). 
 * Output C is [M, 1].
 * For simplicity in hardware, B is loaded entirely into an internal memory (BRAM) first.
 * Then A streams in, and we compute the dot product for each row of A using MAC (Multiply-Accumulate).
 */
case class MatmulOp[T <: Data](dataType: HardType[T], shapeA: Seq[Int], shapeB: Seq[Int], lanes: Int) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  require(shapeB(0) == K && shapeB(1) == 1, "Currently only Matrix-Vector multiplication is supported for simplicity")
  require(K % lanes == 0, "K dimension must be a multiple of lanes")
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, lanes))
    val c = master(Tensor(dataType, Seq(M, 1), lanes = 1)) // Output is a column vector, 1 element per cycle
  }
  
  // 1. Internal Memory (SRAM) for matrix B (the weights)
  val chunksB = K / lanes
  val memB = Mem(Vec(dataType, lanes), chunksB)
  
  // Counters for addressing
  val writeCounterB = Counter(chunksB)
  val readCounterA = Counter(chunksB)
  
  // Accumulator register for the MAC operation
  val accumulator = Reg(dataType)
  accumulator.init(accumulator.getZero)
  // Needs to be cleared to 0 at the start of each row
  
  // Default values to avoid SpinalHDL "NO DRIVER" errors
  io.a.stream.ready := False
  io.b.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload(0).assignFromBits(B(0, widthOf(dataType) bits))
  
  // State Machine (FSM)
  val fsm = new StateMachine {
    val stateLoadB: State = new State with EntryPoint {
      whenIsActive {
        io.b.stream.ready := True
        when(io.b.stream.valid) {
          memB.write(writeCounterB.value, io.b.stream.payload)
          writeCounterB.increment()
          when(writeCounterB.willOverflowIfInc) {
            goto(stateCompute)
          }
        }
      }
    }
    
    val stateCompute: State = new State {
      whenIsActive {
        // We are ready to read A
        io.a.stream.ready := True
        
        // Read B from memory asynchronously for the MAC
        val bData = memB.readAsync(readCounterA.value)
        
        when(io.a.stream.valid) {
          readCounterA.increment()
          
          // MAC calculation (Multiply-Accumulate) for the current lane chunk
          var partialSum: Data = null
          for (i <- 0 until lanes) {
            val mult = (io.a.stream.payload(i), bData(i)) match {
              case (valA: SInt, valB: SInt) => (valA * valB).resized.asInstanceOf[T]
              case (valA: UInt, valB: UInt) => (valA * valB).resized.asInstanceOf[T]
              case _ => throw new Exception("Type unsupported")
            }
            if (partialSum == null) partialSum = mult
            else {
              partialSum = (partialSum, mult) match {
                case (p: SInt, m: SInt) => (p + m).resized.asInstanceOf[T]
                case (p: UInt, m: UInt) => (p + m).resized.asInstanceOf[T]
              }
            }
          }
          
          // Accumulate
          val nextAcc = (accumulator, partialSum) match {
            case (acc: SInt, sum: SInt) => (acc + sum).resized.asInstanceOf[T]
            case (acc: UInt, sum: UInt) => (acc + sum).resized.asInstanceOf[T]
          }
          accumulator := nextAcc
          
          // When a row of A is completely processed (readCounterA overflows)
          when(readCounterA.willOverflowIfInc) {
            goto(stateOutput)
          }
        }
      }
    }
    
    val stateOutput: State = new State {
      whenIsActive {
        // Output the accumulated result
        io.c.stream.valid := True
        io.c.stream.payload(0) := accumulator
        
        when(io.c.stream.ready) {
          // Result consumed, clear accumulator and go back to computing the next row
          accumulator := accumulator.getZero
          goto(stateCompute)
        }
      }
    }
  }
}

object matmul {
  /**
   * Matrix-Vector multiplication using internal BRAM for weights.
   */
  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    require(a.shape.length == 2 && b.shape.length == 2, "Matmul requires 2D tensors")
    require(a.shape(1) == b.shape(0), "Inner dimensions must match (A.cols == B.rows)")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    
    val matmulComp = MatmulOp(a.dataType, a.shape, b.shape, a.lanes)
    matmulComp.io.a <> a
    matmulComp.io.b <> b
    matmulComp.io.c
  }
}
