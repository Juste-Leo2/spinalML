package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

// Transpose 2D tensor [M, N] to [N, M]
// Assumes input stream provides elements row by row
case class TransposeOp[T <: Data](dataType: HardType[T], M: Int, N: Int, lanes: Int) extends Component {
  require(M > 0 && N > 0, "Dimensions must be > 0")
  require((M * N) % lanes == 0, "Total elements must be multiple of lanes")
  require(lanes == 1, "Transpose currently only supports lanes = 1 due to memory alignment constraints")

  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(M, N), lanes))
    val c = master(Tensor(dataType, Seq(N, M), lanes))
  }
  
  val totalElements = M * N
  val mem = Mem(dataType, totalElements)
  
  val writeAddr = Reg(UInt(log2Up(totalElements) bits)) init(0)
  val readRow = Reg(UInt(log2Up(M) bits)) init(0)
  val readCol = Reg(UInt(log2Up(N) bits)) init(0)
  
  io.a.stream.ready := False
  
  // Pipeline for synchronous read (BRAM inference)
  val readAddrStream = Stream(UInt(log2Up(totalElements) bits))
  readAddrStream.valid := False
  readAddrStream.payload := (readRow * N) + readCol
  
  val readData = mem.readSync(readAddrStream.payload.resized, enable = readAddrStream.ready)
  
  val outValid = RegInit(False)
  when(readAddrStream.ready) {
    outValid := readAddrStream.valid
  }
  
  readAddrStream.ready := io.c.stream.ready || !outValid
  io.c.stream.valid := outValid
  io.c.stream.payload(0) := readData
  
  val fsm = new StateMachine {
    val stateWrite: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          mem.write(writeAddr, io.a.stream.payload(0))
          writeAddr := writeAddr + 1
          when(writeAddr === totalElements - 1) {
            goto(stateWaitFlush)
          }
        }
      }
    }
    
    val stateWaitFlush: State = new State {
       whenIsActive {
          // Just 1 cycle delay to ensure write completes if needed, though BRAM write is 1 cycle.
          goto(stateRead)
       }
    }
    
    val stateRead: State = new State {
      whenIsActive {
        readAddrStream.valid := True
        when(readAddrStream.ready) {
          readRow := readRow + 1
          when(readRow === M - 1) {
            readRow := 0
            readCol := readCol + 1
            when(readCol === N - 1) {
              readCol := 0
              writeAddr := 0
              goto(stateWaitEmpty)
            }
          }
        }
      }
    }
    
    val stateWaitEmpty: State = new State {
      whenIsActive {
         // Wait for the pipeline to empty before returning to write
         when(!outValid || io.c.stream.ready) {
            goto(stateWrite)
         }
      }
    }
  }
}

object transpose {
  def apply[T <: Data](a: Tensor[T]): Tensor[T] = {
    require(a.shape.length == 2, "Transpose currently only supports 2D tensors")
    val M = a.shape(0)
    val N = a.shape(1)
    val comp = TransposeOp(a.dataType, M, N, a.lanes)
    comp.io.a <> a
    comp.io.c
  }
}
