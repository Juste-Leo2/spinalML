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
case class Im2ColOp[T <: Data](dataType: HardType[T], H: Int, W: Int, C: Int, K: Int, outLanes: Int) extends Component {
  require(H >= K && W >= K, "Image dimensions must be >= kernel size")
  val H_out = H - K + 1
  val W_out = W - K + 1
  val totalWindows = H_out * W_out
  val windowSize = K * K * C
  require(windowSize % outLanes == 0, "Window size must be divisible by outLanes")
  val outCycles = windowSize / outLanes
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(H, W, C), lanes = 1))
    val c = master(Tensor(dataType, Seq(totalWindows, windowSize), lanes = outLanes))
  }
  
  // Line Buffers to hold previous rows of the image.
  // Each row has W pixels, and each pixel has C channels.
  val lineBuffers = for (i <- 0 until K - 1) yield new Area {
    val regs = Vec(Reg(dataType), W * C)
    regs.foreach(r => r.init(r.getZero.asInstanceOf[T]))
    val pop = regs(W * C - 1) // Oldest element
  }
  
  // 1D Shift Register holding the flattened window in row-major order [K, K, C]
  val shiftReg = Vec(Reg(dataType), windowSize)
  shiftReg.foreach(r => r.init(r.getZero.asInstanceOf[T]))
  
  // Temporary buffers to hold a single temporal column of K pixels (each has C channels)
  // row 0 is oldest (from line buffer K-2), row K-1 is newest (from input)
  val tempVecs = Vec.fill(K)(Vec(Reg(dataType), C))
  for (r <- 0 until K; c <- 0 until C) tempVecs(r)(c).init(tempVecs(r)(c).getZero.asInstanceOf[T])
  
  val channelCount = Counter(C)
  val x = Counter(W)
  val y = Counter(H)
  val windowCount = Counter(totalWindows)
  val outChunkCount = Counter(outCycles)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Map output payload directly from shift register based on current chunk
  for(i <- 0 until outLanes) {
    val flatIndex = (outChunkCount.value * outLanes) + i
    io.c.stream.payload(i) := shiftReg(flatIndex.resized)
  }
  
  val isWindowValid = (x.value >= (K - 1)) && (y.value >= (K - 1))
  
  val fsm = new StateMachine {
    
    // Combinational vectors for the full column of K pixels (each C channels)
    // We use tempVecs (registers) for 0 to C-2, and the current payload/pop for C-1
    val currentPixels = Vec.fill(K)(Vec(dataType, C))
    for (r <- 0 until K) {
      for (ch <- 0 until C - 1) {
        currentPixels(r)(ch) := tempVecs(r)(ch)
      }
      if (r == K - 1) {
        currentPixels(r)(C - 1) := io.a.stream.payload(0)
      } else {
        currentPixels(r)(C - 1) := lineBuffers(K - 2 - r).pop
      }
    }
    
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          // Push to tempVecs (only matters for C > 1)
          tempVecs(K - 1)(channelCount.value) := io.a.stream.payload(0)
          for (i <- 0 until K - 1) {
            tempVecs(K - 2 - i)(channelCount.value) := lineBuffers(i).pop
          }
          
          // Shift Line Buffers
          if (K > 1) {
            for (i <- (1 until K - 1).reverse) {
              for (j <- (1 until W * C).reverse) lineBuffers(i).regs(j) := lineBuffers(i).regs(j - 1)
              lineBuffers(i).regs(0) := lineBuffers(i - 1).pop
            }
            for (j <- (1 until W * C).reverse) lineBuffers(0).regs(j) := lineBuffers(0).regs(j - 1)
            lineBuffers(0).regs(0) := io.a.stream.payload(0)
          }
          
          channelCount.increment()
          when(channelCount.willOverflowIfInc) {
            // Shift the 2D window by 1 column (C elements) for each row
            for (r <- 0 until K) {
              val rowOffset = r * K * C
              for (i <- 0 until K * C - C) {
                shiftReg(rowOffset + i) := shiftReg(rowOffset + i + C)
              }
              for (ch <- 0 until C) {
                shiftReg(rowOffset + K * C - C + ch) := currentPixels(r)(ch)
              }
            }
            
            x.increment()
            when(x.willOverflowIfInc) { y.increment() }
            
            when(isWindowValid) {
              goto(stateOutput)
            }
          }
        }
      }
    }
    
    val stateOutput: State = new State {
      whenIsActive {
        io.c.stream.valid := True
        
        when(io.c.stream.ready) {
          outChunkCount.increment()
          when(outChunkCount.willOverflowIfInc) {
            windowCount.increment()
            when(windowCount.willOverflowIfInc) {
              goto(stateDone)
            } otherwise {
              // K = 1: every input pixel is a window, and stateFill already
              // shifts + counts exactly one pixel per beat — forwarding to
              // stateWaitA would absorb a second pixel per window and emit
              // every other one (the M3 window-count anomaly).
              if (K == 1) goto(stateFill) else goto(stateWaitA)
            }
          }
        }
      }
    }
    
    val stateWaitA: State = new State {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          tempVecs(K - 1)(channelCount.value) := io.a.stream.payload(0)
          for (i <- 0 until K - 1) {
            tempVecs(K - 2 - i)(channelCount.value) := lineBuffers(i).pop
          }
          
          if (K > 1) {
            for (i <- (1 until K - 1).reverse) {
              for (j <- (1 until W * C).reverse) lineBuffers(i).regs(j) := lineBuffers(i).regs(j - 1)
              lineBuffers(i).regs(0) := lineBuffers(i - 1).pop
            }
            for (j <- (1 until W * C).reverse) lineBuffers(0).regs(j) := lineBuffers(0).regs(j - 1)
            lineBuffers(0).regs(0) := io.a.stream.payload(0)
          }
          
          channelCount.increment()
          when(channelCount.willOverflowIfInc) {
            for (r <- 0 until K) {
              val rowOffset = r * K * C
              for (i <- 0 until K * C - C) {
                shiftReg(rowOffset + i) := shiftReg(rowOffset + i + C)
              }
              for (ch <- 0 until C) {
                shiftReg(rowOffset + K * C - C + ch) := currentPixels(r)(ch)
              }
            }
            
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
    }
    
    val stateDone: State = new State {
       whenIsActive {
         x.clear()
         y.clear()
         channelCount.clear()
         windowCount.clear()
         outChunkCount.clear()
         goto(stateFill)
       }
    }
  }
}

object im2col {
  def apply[T <: Data](a: Tensor[T], kernelSize: Int, outLanes: Int): Tensor[T] = {
    val C = if (a.shape.length == 3) a.shape(2) else 1
    require(a.lanes == 1, "Im2Col input must have lanes = 1")
    
    val comp = Im2ColOp(a.dataType, a.shape(0), a.shape(1), C, kernelSize, outLanes)
    comp.io.a <> a
    comp.io.c
  }
}
