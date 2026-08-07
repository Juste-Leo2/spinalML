package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

/**
 * LineBuffer: A simple memory that delays data by exactly `depth` valid cycles.
 * Used to store rows of an image for 2D convolutions.
 */
case class LineBuffer[T <: Data](dataType: HardType[T], depth: Int) extends Component {
  val io = new Bundle {
    val push = in(dataType())
    val pop = out(dataType())
    val en = in Bool()
  }
  
  // If depth is very small (like for small tests), use registers.
  // Otherwise use Mem (BRAM). Mem(..., depth) works nicely.
  val mem = Mem(dataType, depth)
  val ptr = Counter(depth)
  
  // Read oldest data
  io.pop := mem.readSync(ptr.value) // Using Sync read is better for BRAM, but requires 1 cycle delay.
  // Wait, if we use readSync, the output is delayed by 1 cycle.
  // If we readAsync, it's combinational, but BRAM doesn't support async read.
  // Let's use registers for simplicity in this abstract ML framework, unless depth is huge.
  // SpinalHDL Mem will infer registers if readAsync is used.
}

/**
 * Im2ColOp: Converts a 2D image into flattened sliding windows.
 * Input A: shape [H, W], lanes = 1
 * Output C: shape [H_out * W_out, K * K], lanes = K * K
 */
case class Im2ColOp[T <: Data](dataType: HardType[T], H: Int, W: Int, K: Int) extends Component {
  require(H >= K && W >= K, "Image dimensions must be >= kernel size")
  val H_out = H - K + 1
  val W_out = W - K + 1
  val totalWindows = H_out * W_out
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(H, W), lanes = 1))
    val c = master(Tensor(dataType, Seq(totalWindows, K * K), lanes = K * K))
  }
  
  // Register-based Line Buffers for simplicity and async read (perfect for any W)
  val lineBuffers = for (i <- 0 until K - 1) yield new Area {
    val regs = Vec(Reg(dataType), W)
    regs.foreach(r => r.init(r.getZero.asInstanceOf[T]))
    val pop = regs(W - 1) // Oldest element
  }
  
  // 2D Shift Register for the sliding window: window(row)(col)
  // row 0 is the oldest row (from the last line buffer).
  // row K-1 is the newest row (from incoming data).
  // col 0 is the newest column in that row.
  // col K-1 is the oldest column.
  val window = Vec.fill(K)(Vec.fill(K)(Reg(dataType)))
  for (r <- 0 until K) {
    for (c <- 0 until K) {
      window(r)(c).init(window(r)(c).getZero.asInstanceOf[T])
    }
  }
  
  val x = Counter(W)
  val y = Counter(H)
  val windowCount = Counter(totalWindows)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Flatten the window to output payload
  // Standard PyTorch order: row by row, then col by col.
  for (r <- 0 until K) {
    for (c <- 0 until K) {
      // Oldest elements first (r=0, c=K-1)
      val flatIndex = r * K + c
      io.c.stream.payload(flatIndex) := window(r)(K - 1 - c)
    }
  }
  
  val isWindowValid = (x.value >= (K - 1)) && (y.value >= (K - 1))
  
  val fsm = new StateMachine {
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          // 1. Update Line Buffers
          if (K > 1) {
            for (i <- (1 until K - 1).reverse) {
              for (j <- (1 until W).reverse) lineBuffers(i).regs(j) := lineBuffers(i).regs(j - 1)
              lineBuffers(i).regs(0) := lineBuffers(i - 1).pop
            }
            for (j <- (1 until W).reverse) lineBuffers(0).regs(j) := lineBuffers(0).regs(j - 1)
            lineBuffers(0).regs(0) := io.a.stream.payload(0)
          }
          
          // 2. Shift the 2D window
          for (r <- 0 until K) {
            for (c <- (1 until K).reverse) {
              window(r)(c) := window(r)(c - 1)
            }
          }
          window(K - 1)(0) := io.a.stream.payload(0)
          for (i <- 0 until K - 1) {
             window(K - 2 - i)(0) := lineBuffers(i).pop
          }
          
          // 3. Update Coordinates
          x.increment()
          when(x.willOverflowIfInc) { y.increment() }
          
          // 4. Check if the CURRENT pixel forms a valid window
          when(isWindowValid) {
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
                 if (K > 1) {
                   for (i <- (1 until K - 1).reverse) {
                     for (j <- (1 until W).reverse) lineBuffers(i).regs(j) := lineBuffers(i).regs(j - 1)
                     lineBuffers(i).regs(0) := lineBuffers(i - 1).pop
                   }
                   for (j <- (1 until W).reverse) lineBuffers(0).regs(j) := lineBuffers(0).regs(j - 1)
                   lineBuffers(0).regs(0) := io.a.stream.payload(0)
                 }
                 for (r <- 0 until K) {
                   for (c <- (1 until K).reverse) window(r)(c) := window(r)(c - 1)
                 }
                 window(K - 1)(0) := io.a.stream.payload(0)
                 for (i <- 0 until K - 1) window(K - 2 - i)(0) := lineBuffers(i).pop
                 
                 x.increment()
                 when(x.willOverflowIfInc) { y.increment() }
                 
                 when(!isWindowValid) {
                   goto(stateFill)
                 }
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
             if (K > 1) {
               for (i <- (1 until K - 1).reverse) {
                 for (j <- (1 until W).reverse) lineBuffers(i).regs(j) := lineBuffers(i).regs(j - 1)
                 lineBuffers(i).regs(0) := lineBuffers(i - 1).pop
               }
               for (j <- (1 until W).reverse) lineBuffers(0).regs(j) := lineBuffers(0).regs(j - 1)
               lineBuffers(0).regs(0) := io.a.stream.payload(0)
             }
             for (r <- 0 until K) {
               for (c <- (1 until K).reverse) window(r)(c) := window(r)(c - 1)
             }
             window(K - 1)(0) := io.a.stream.payload(0)
             for (i <- 0 until K - 1) window(K - 2 - i)(0) := lineBuffers(i).pop
             
             x.increment()
             when(x.willOverflowIfInc) { y.increment() }
             
             when(isWindowValid) {
               goto(stateOutput)
             } otherwise {
               goto(stateFill)
             }
          }
       }
    }
    
    val stateDone: State = new State {
       whenIsActive {
          x.clear()
          y.clear()
          windowCount.clear()
          goto(stateFill)
       }
    }
  }
}

object im2col {
  def apply[T <: Data](a: Tensor[T], kernelSize: Int): Tensor[T] = {
    require(a.shape.length == 2, "Im2Col expects a 2D tensor [H, W]")
    require(a.lanes == 1, "Im2Col input must have lanes = 1")
    
    val comp = Im2ColOp(a.dataType, a.shape(0), a.shape(1), kernelSize)
    comp.io.a <> a
    comp.io.c
  }
}
