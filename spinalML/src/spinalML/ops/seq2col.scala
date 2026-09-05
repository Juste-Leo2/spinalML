// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

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
case class Seq2ColOp[T <: Data](dataType: HardType[T], L: Int, C: Int, K: Int, outLanes: Int) extends Component {
  require(L >= K, "Sequence length L must be greater or equal to kernel size K")
  val L_out = L - K + 1
  val windowSize = K * C
  require(windowSize % outLanes == 0, "Window size must be divisible by outLanes")
  val outCycles = windowSize / outLanes
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(L, C), lanes = 1))
    val c = master(Tensor(dataType, Seq(L_out, windowSize), lanes = outLanes))
  }
  
  // Shift register holding the flattened window.
  // We shift it such that the oldest elements are at the beginning (index 0)
  // and the newest elements are at the end (index windowSize - 1).
  val shiftReg = Vec(Reg(dataType), windowSize)
  shiftReg.foreach(r => r.init(r.getZero.asInstanceOf[T]))
  
  // Temporary buffer to hold a single temporal step (C channels)
  val tempVec = Vec(Reg(dataType), C)
  tempVec.foreach(r => r.init(r.getZero.asInstanceOf[T]))
  
  val elementCount = Counter(L)
  val channelCount = Counter(C)
  val windowCount = Counter(L_out)
  val outChunkCount = Counter(outCycles)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Map output payload directly from shift register based on current chunk
  for(i <- 0 until outLanes) {
    val flatIndex = (outChunkCount.value * outLanes) + i
    io.c.stream.payload(i) := shiftReg(flatIndex.resized)
  }
  
  val fsm = new StateMachine {
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          tempVec(channelCount.value) := io.a.stream.payload(0)
          channelCount.increment()
          
          when(channelCount.willOverflowIfInc) {
            // Shift the register by C positions
            for (i <- 0 until windowSize - C) {
              shiftReg(i) := shiftReg(i + C)
            }
            // Insert newest temporal step at the end
            for (c <- 0 until C - 1) {
              shiftReg(windowSize - C + c) := tempVec(c)
            }
            shiftReg(windowSize - 1) := io.a.stream.payload(0)
            
            elementCount.increment()
            when(elementCount.value === K - 1) {
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
          tempVec(channelCount.value) := io.a.stream.payload(0)
          channelCount.increment()
          
          when(channelCount.willOverflowIfInc) {
            for (i <- 0 until windowSize - C) {
              shiftReg(i) := shiftReg(i + C)
            }
            for (c <- 0 until C - 1) {
              shiftReg(windowSize - C + c) := tempVec(c)
            }
            shiftReg(windowSize - 1) := io.a.stream.payload(0)
            
            goto(stateOutput)
          }
        }
      }
    }
    
    val stateDone: State = new State {
       whenIsActive {
         elementCount.clear()
         windowCount.clear()
         channelCount.clear()
         outChunkCount.clear()
         goto(stateFill)
       }
    }
  }
}

object seq2col {
  def apply[T <: Data](a: Tensor[T], kernelSize: Int, outLanes: Int): Tensor[T] = {
    require(a.shape.length == 2, "Seq2Col expects a 1D tensor [L, C]")
    require(a.lanes == 1, "Seq2Col input must have lanes = 1")
    
    val comp = Seq2ColOp(a.dataType, a.shape(0), a.shape(1), kernelSize, outLanes)
    comp.io.a <> a
    comp.io.c
  }
}
