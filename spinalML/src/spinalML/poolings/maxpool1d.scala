// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.poolings

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

case class MaxPool1DOp[T <: Data](dataType: HardType[T], L: Int, channels: Int, poolSize: Int, stride: Int) extends Component {
  require(L >= poolSize, "Sequence length L must be >= poolSize")
  val L_out = (L - poolSize) / stride + 1
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(L, channels), lanes = channels))
    val c = master(Tensor(dataType, Seq(L_out, channels), lanes = channels))
  }
  
  // Shift registers to hold the window for the max operation for each channel
  val shiftRegs = Seq.fill(channels)(Vec(Reg(dataType), poolSize))
  shiftRegs.foreach(_.foreach(r => r.init(r.getZero.asInstanceOf[T])))
  
  val elementCount = Counter(L)
  val windowCount = Counter(L_out)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Combinatorial max computation (Max-Tree)
  def buildMaxTree(nodes: Seq[T]): T = {
    if (nodes.length == 1) return nodes.head
    val nextLevel = nodes.grouped(2).map {
      case Seq(a) => a
      case Seq(a, b) =>
        (a, b) match {
          case (valA: SInt, valB: SInt) => Mux(valA > valB, valA, valB).asInstanceOf[T]
          case (valA: UInt, valB: UInt) => Mux(valA > valB, valA, valB).asInstanceOf[T]
          case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.max(valA, valB).asInstanceOf[T]
          case _ => throw new Exception("Data type not supported for MaxPool")
        }
    }.toSeq
    buildMaxTree(nextLevel)
  }
  
  for (ch <- 0 until channels) {
    val currentMax = buildMaxTree(shiftRegs(ch).toSeq)
    io.c.stream.payload(ch) := currentMax
  }
  
  val fsm = new StateMachine {
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          for (ch <- 0 until channels) {
            for (i <- (1 until poolSize).reverse) {
              shiftRegs(ch)(i) := shiftRegs(ch)(i - 1)
            }
            shiftRegs(ch)(0) := io.a.stream.payload(ch)
          }
          
          elementCount.increment()
          when(elementCount.value === poolSize - 1) {
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
             goto(stateSlide)
          }
        }
      }
    }
    
    val slideCounter = Counter(stride)
    
    val stateSlide: State = new State {
      onEntry {
        slideCounter.clear()
      }
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          for (ch <- 0 until channels) {
            for (i <- (1 until poolSize).reverse) {
              shiftRegs(ch)(i) := shiftRegs(ch)(i - 1)
            }
            shiftRegs(ch)(0) := io.a.stream.payload(ch)
          }
          
          slideCounter.increment()
          elementCount.increment()
          when(slideCounter.value === stride - 1) {
            goto(stateOutput)
          }
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

object maxpool1d {
  def apply[T <: Data](a: Tensor[T], poolSize: Int, stride: Int): Tensor[T] = {
    require(a.shape.length == 2, "MaxPool1D expects a 2D tensor [L, channels]")
    val channels = a.shape(1)
    require(a.lanes == channels, s"MaxPool1D input must have lanes = channels ($channels)")
    
    val comp = MaxPool1DOp(a.dataType, a.shape(0), channels, poolSize, stride)
    comp.io.a <> a
    comp.io.c
  }
}
