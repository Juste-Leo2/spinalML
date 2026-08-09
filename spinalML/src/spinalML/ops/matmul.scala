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
    val c = master(Tensor(accType, Seq(M, 1), lanes = 1))
  }
  
  val chunksTile = tileSize / lanes
  
  val bufferB = StreamDoubleBuffer(dataType, tileSize, lanes)
  bufferB.io.streamIn << io.b.stream
  
  val readCounterTile = Counter(chunksTile)
  
  val accumulators = Vec(Reg(accType), M)
  accumulators.foreach(acc => acc.init(acc.getZero.asInstanceOf[TAcc]))
  
  val rowCounter = Counter(M)
  val tileCounter = Counter(K / tileSize)
  
  bufferB.io.readAddr := readCounterTile.value
  bufferB.io.nextTile := False
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload(0).assignFromBits(B(0, widthOf(accType) bits))
  
  // Pipeline Signals
  val stage1_fire = False
  
  val stage2_valid = RegNext(stage1_fire, init = False)
  val stage2_row = RegNextWhen(rowCounter.value, stage1_fire)
  val stage2_a = RegNextWhen(io.a.stream.payload, stage1_fire)
  
  val stage3_valid = RegNext(stage2_valid, init = False)
  val stage3_row = RegNextWhen(stage2_row, stage2_valid)
  
  val multRegs = Vec(Reg(accType), lanes)
  multRegs.foreach(r => r.init(r.getZero.asInstanceOf[TAcc]))
  
  // Stage 2: Multiplications (and sync read arrives from bufferB)
  when(stage2_valid) {
    for (i <- 0 until lanes) {
      multRegs(i) := ((stage2_a(i), bufferB.io.readData(i)) match {
        case (valA: SInt, valB: SInt) => (valA * valB).resized.asInstanceOf[TAcc]
        case (valA: UInt, valB: UInt) => (valA * valB).resized.asInstanceOf[TAcc]
        case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.mul(valA, valB).asInstanceOf[TAcc]
        case _ => throw new Exception("Type unsupported")
      })
    }
  }
  
  // Stage 3: Addition tree and accumulation
  when(stage3_valid) {
    var partialSum: Data = null
    for (i <- 0 until lanes) {
      if (partialSum == null) partialSum = multRegs(i)
      else {
        partialSum = ((partialSum, multRegs(i)) match {
          case (p: SInt, m: SInt) => (p + m).resized.asInstanceOf[TAcc]
          case (p: UInt, m: UInt) => (p + m).resized.asInstanceOf[TAcc]
          case (p: spinalML.dtypes.FloatML, m: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(p, m).asInstanceOf[TAcc]
        })
      }
    }
    
    val currentAcc = accumulators(stage3_row)
    val nextAcc = ((currentAcc, partialSum) match {
      case (acc: SInt, sum: SInt) => (acc + sum).resized.asInstanceOf[TAcc]
      case (acc: UInt, sum: UInt) => (acc + sum).resized.asInstanceOf[TAcc]
      case (acc: spinalML.dtypes.FloatML, sum: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(acc, sum).asInstanceOf[TAcc]
    })
    accumulators(stage3_row) := nextAcc
  }
  
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
        io.a.stream.ready := True
        
        when(io.a.stream.valid) {
          stage1_fire := True
          
          readCounterTile.increment()
          when(readCounterTile.willOverflowIfInc) {
            rowCounter.increment()
            when(rowCounter.willOverflowIfInc) {
              bufferB.io.nextTile := True
              tileCounter.increment()
              when(tileCounter.willOverflowIfInc) {
                goto(stateWaitFlushEnd)
              } otherwise {
                goto(stateWaitFlushTile)
              }
            }
          }
        }
      }
    }
    
    val stateWaitFlushTile: State = new State {
      whenIsActive {
        io.a.stream.ready := False
        when(!stage2_valid && !stage3_valid) {
          goto(stateWaitTile)
        }
      }
    }
    
    val stateWaitFlushEnd: State = new State {
      whenIsActive {
        io.a.stream.ready := False
        when(!stage2_valid && !stage3_valid) {
          goto(stateOutput)
        }
      }
    }
    
    val outCounter = Counter(M)
    
    val stateOutput: State = new State {
      whenIsActive {
        io.c.stream.valid := True
        io.c.stream.payload(0) := accumulators(outCounter.value)
        
        when(io.c.stream.ready) {
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
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], b: Tensor[T], accType: HardType[TAcc], tileSize: Int): Tensor[TAcc] = {
    require(a.shape.length == 2 && b.shape.length == 2, "Matmul requires 2D tensors")
    require(a.shape(1) == b.shape(0), "Inner dimensions must match (A.cols == B.rows)")
    require(a.lanes == b.lanes, "Tensors must have the same lanes")
    
    val matmulComp = MatmulOp(a.dataType, accType, a.shape, b.shape, a.lanes, tileSize)
    matmulComp.io.a <> a
    matmulComp.io.b <> b
    matmulComp.io.c
  }

  def apply[T <: Data](a: Tensor[T], b: Tensor[T], tileSize: Int = 1024): Tensor[T] = {
    apply(a, b, a.dataType, tileSize)
  }
}
