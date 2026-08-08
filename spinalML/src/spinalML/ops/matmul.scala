package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor
import spinalML.memory.StreamDoubleBuffer

/**
 * MatmulOp: Matrix-Vector multiplication using Double-Buffering (Ping-Pong).
 * A is [M, K], B is [K, 1] (Vector). 
 * Output C is [M, 1].
 * Option A implementation: B is loaded in tiles (tileSize). A is streamed in column-blocks.
 */
case class MatmulOp[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc], shapeA: Seq[Int], shapeB: Seq[Int], lanes: Int, tileSize: Int = 1024) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  require(shapeB(0) == K && shapeB(1) == 1, "Currently only Matrix-Vector multiplication is supported for simplicity")
  require(K % lanes == 0, "K dimension must be a multiple of lanes")
  require(tileSize % lanes == 0, "tileSize must be a multiple of lanes")
  require(K % tileSize == 0, "K dimension must be a multiple of tileSize")
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, lanes))
    val c = master(Tensor(accType, Seq(M, 1), lanes = 1)) // Output is a column vector, 1 element per cycle
  }
  
  val chunksTile = tileSize / lanes
  
  // 1. Double Buffer for weights B
  val bufferB = StreamDoubleBuffer(dataType, tileSize, lanes)
  bufferB.io.streamIn << io.b.stream
  
  // Counters for addressing
  val readCounterTile = Counter(chunksTile)
  
  // Accumulators for the MAC operation (Option A: M accumulators for partial sums)
  val accumulators = Vec(Reg(accType), M)
  accumulators.foreach(acc => acc.init(acc.getZero.asInstanceOf[TAcc]))
  
  val rowCounter = Counter(M)
  val tileCounter = Counter(K / tileSize)
  
  // Wiring buffer
  bufferB.io.readAddr := readCounterTile.value
  bufferB.io.nextTile := False
  
  // Default values to avoid SpinalHDL "NO DRIVER" errors
  io.a.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload(0).assignFromBits(B(0, widthOf(accType) bits))
  
  // State Machine (FSM)
  val fsm = new StateMachine {
    val stateWaitTile: State = new State with EntryPoint {
      whenIsActive {
        when(bufferB.io.tileReady) {
          goto(stateComputeTile)
        }
      }
    }
    
    val stateComputeTile: State = new State {
      whenIsActive {
        // We are ready to read A
        io.a.stream.ready := True
        
        // Read B from memory asynchronously for the MAC
        val bData = bufferB.io.readData
        
        when(io.a.stream.valid) {
          
          // MAC calculation (Multiply-Accumulate) for the current lane chunk
          var partialSum: Data = null
          for (i <- 0 until lanes) {
            val mult = (io.a.stream.payload(i), bData(i)) match {
              case (valA: SInt, valB: SInt) => (valA * valB).resized.asInstanceOf[TAcc]
              case (valA: UInt, valB: UInt) => (valA * valB).resized.asInstanceOf[TAcc]
              case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.mul(valA, valB).asInstanceOf[TAcc]
              case _ => throw new Exception("Type unsupported")
            }
            if (partialSum == null) partialSum = mult
            else {
              partialSum = (partialSum, mult) match {
                case (p: SInt, m: SInt) => (p + m).resized.asInstanceOf[TAcc]
                case (p: UInt, m: UInt) => (p + m).resized.asInstanceOf[TAcc]
                case (p: spinalML.dtypes.FloatML, m: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(p, m).asInstanceOf[TAcc]
              }
            }
          }
          
          // Accumulate with the current row's partial sum
          val currentAcc = accumulators(rowCounter.value)
          val nextAcc = (currentAcc, partialSum) match {
            case (acc: SInt, sum: SInt) => (acc + sum).resized.asInstanceOf[TAcc]
            case (acc: UInt, sum: UInt) => (acc + sum).resized.asInstanceOf[TAcc]
            case (acc: spinalML.dtypes.FloatML, sum: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(acc, sum).asInstanceOf[TAcc]
          }
          accumulators(rowCounter.value) := nextAcc
          
          readCounterTile.increment()
          when(readCounterTile.willOverflowIfInc) {
            // End of tile for this row
            rowCounter.increment()
            when(rowCounter.willOverflowIfInc) {
              // End of tile for ALL rows -> Next tile
              bufferB.io.nextTile := True
              tileCounter.increment()
              when(tileCounter.willOverflowIfInc) {
                // End of all tiles! Output result
                goto(stateOutput)
              } otherwise {
                goto(stateWaitTile)
              }
            }
          }
        }
      }
    }
    
    val outCounter = Counter(M)
    
    val stateOutput: State = new State {
      whenIsActive {
        // Output the accumulated results
        io.c.stream.valid := True
        io.c.stream.payload(0) := accumulators(outCounter.value)
        
        when(io.c.stream.ready) {
          // Result consumed, clear accumulator and go to next
          accumulators(outCounter.value) := accumulators(0).getZero.asInstanceOf[TAcc]
          outCounter.increment()
          when(outCounter.willOverflowIfInc) {
             goto(stateWaitTile)
          }
        }
      }
    }
  }
}

object matmul {
  /**
   * Matrix-Vector multiplication using Double Buffered BRAM for weights.
   * Allows specifying a different accumulator type to prevent overflow.
   */
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], b: Tensor[T], accType: HardType[TAcc], tileSize: Int): Tensor[TAcc] = {
    require(a.shape.length == 2 && b.shape.length == 2, "Matmul requires 2D tensors")
    require(a.shape(1) == b.shape(0), "Inner dimensions must match (A.cols == B.rows)")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    
    val matmulComp = MatmulOp(a.dataType, accType, a.shape, b.shape, a.lanes, tileSize)
    matmulComp.io.a <> a
    matmulComp.io.b <> b
    matmulComp.io.c
  }

  /**
   * Helper where accumulator type defaults to input type.
   */
  def apply[T <: Data](a: Tensor[T], b: Tensor[T], tileSize: Int = 1024): Tensor[T] = {
    apply(a, b, a.dataType, tileSize)
  }
}

