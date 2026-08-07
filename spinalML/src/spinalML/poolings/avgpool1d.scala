package spinalML.poolings

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

case class AvgPool1DOp[T <: Data](dataType: HardType[T], L: Int, poolSize: Int, stride: Int) extends Component {
  require(L >= poolSize, "Sequence length L must be >= poolSize")
  require(isPow2(poolSize), "poolSize must be a power of 2 for AvgPool (shift based)")
  
  val L_out = (L - poolSize) / stride + 1
  val shift = log2Up(poolSize)
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(L, 1), lanes = 1))
    val c = master(Tensor(dataType, Seq(L_out, 1), lanes = 1))
  }
  
  val shiftReg = Vec(Reg(dataType), poolSize)
  shiftReg.foreach(r => r.init(r.getZero.asInstanceOf[T]))
  
  val elementCount = Counter(L)
  val windowCount = Counter(L_out)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Combinatorial average computation (sum then shift)
  io.c.stream.payload(0) match {
    case _: SInt => 
      var acc = shiftReg(0).asInstanceOf[SInt].resize(dataType.getBitsWidth + shift)
      for (i <- 1 until poolSize) {
        acc = acc + shiftReg(i).asInstanceOf[SInt].resize(dataType.getBitsWidth + shift)
      }
      io.c.stream.payload(0).assignFrom((acc >> shift).resize(dataType.getBitsWidth).asInstanceOf[T])
    case _: UInt =>
      var acc = shiftReg(0).asInstanceOf[UInt].resize(dataType.getBitsWidth + shift)
      for (i <- 1 until poolSize) {
        acc = acc + shiftReg(i).asInstanceOf[UInt].resize(dataType.getBitsWidth + shift)
      }
      io.c.stream.payload(0).assignFrom((acc >> shift).resize(dataType.getBitsWidth).asInstanceOf[T])
    case _ => 
      throw new Exception("Data type not supported for AvgPool")
  }
  
  val fsm = new StateMachine {
    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          for (i <- (1 until poolSize).reverse) {
            shiftReg(i) := shiftReg(i - 1)
          }
          shiftReg(0) := io.a.stream.payload(0)
          
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
          for (i <- (1 until poolSize).reverse) {
            shiftReg(i) := shiftReg(i - 1)
          }
          shiftReg(0) := io.a.stream.payload(0)
          
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

object avgpool1d {
  def apply[T <: Data](a: Tensor[T], poolSize: Int, stride: Int): Tensor[T] = {
    require(a.shape.length == 2 && a.shape(1) == 1, "AvgPool1D expects a 1D tensor [L, 1]")
    require(a.lanes == 1, "AvgPool1D input must have lanes = 1")
    
    val comp = AvgPool1DOp(a.dataType, a.shape(0), poolSize, stride)
    comp.io.a <> a
    comp.io.c
  }
}
