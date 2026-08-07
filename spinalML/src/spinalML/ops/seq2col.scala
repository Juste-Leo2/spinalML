package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

/**
 * Seq2ColOp (1D Im2Col): Converts a 1D sequence into sliding windows.
 * Input A: shape [L, 1], lanes = 1 (streams 1 element per cycle)
 * Output C: shape [L - K + 1, K], lanes = K (streams a full window of size K per cycle)
 */
case class Seq2ColOp[T <: Data](dataType: HardType[T], L: Int, K: Int) extends Component {
  require(L >= K, "Sequence length L must be greater or equal to kernel size K")
  val L_out = L - K + 1
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(L, 1), lanes = 1))
    val c = master(Tensor(dataType, Seq(L_out, K), lanes = K))
  }
  
  // Shift register to hold the window (index 0 is newest, K-1 is oldest)
  val shiftReg = Vec(Reg(dataType), K)
  shiftReg.foreach(r => r.init(r.getZero.asInstanceOf[T]))
  
  val elementCount = Counter(L)
  val windowCount = Counter(L_out)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  for(i <- 0 until K) {
    // Output oldest first
    io.c.stream.payload(i) := shiftReg(K - 1 - i)
  }
  
  val fsm = new StateMachine {
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          for (i <- (1 until K).reverse) {
            shiftReg(i) := shiftReg(i - 1)
          }
          shiftReg(0) := io.a.stream.payload(0)
          
          elementCount.increment()
          when(elementCount.value === K - 1) {
            goto(stateOutput)
          }
        }
      }
    }
    
    val stateOutput: State = new State {
      whenIsActive {
        io.c.stream.valid := True
        
        when(io.c.stream.ready) {
          windowCount.increment()
          when(windowCount.willOverflowIfInc) {
             goto(stateDone)
          } otherwise {
             io.a.stream.ready := True
             when(io.a.stream.valid) {
               for (i <- (1 until K).reverse) {
                 shiftReg(i) := shiftReg(i - 1)
               }
               shiftReg(0) := io.a.stream.payload(0)
             } otherwise {
               goto(stateWaitA)
             }
          }
        }
      }
    }
    
    val stateWaitA: State = new State {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          for (i <- (1 until K).reverse) {
            shiftReg(i) := shiftReg(i - 1)
          }
          shiftReg(0) := io.a.stream.payload(0)
          goto(stateOutput)
        }
      }
    }
    
    val stateDone: State = new State {
       whenIsActive {
         elementCount.clear()
         windowCount.clear()
         goto(stateFill)
       }
    }
  }
}

object seq2col {
  def apply[T <: Data](a: Tensor[T], kernelSize: Int): Tensor[T] = {
    require(a.shape.length == 2 && a.shape(1) == 1, "Seq2Col expects a 1D tensor [L, 1]")
    require(a.lanes == 1, "Seq2Col input must have lanes = 1")
    
    val comp = Seq2ColOp(a.dataType, a.shape(0), kernelSize)
    comp.io.a <> a
    comp.io.c
  }
}
