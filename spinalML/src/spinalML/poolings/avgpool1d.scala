// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.poolings

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor
import spinalML.ops.{repack, RequantizeMath}
import spinalML.{RoundingConfig, RoundingMode}

case class AvgPool1DOp[T <: Data](dataType: HardType[T], L: Int, channels: Int, poolSize: Int, stride: Int,
                                  rounding: RoundingMode = RoundingConfig.current) extends Component {
  require(L >= poolSize, "Sequence length L must be >= poolSize")
  require(isPow2(poolSize), "poolSize must be a power of 2 for AvgPool (shift based)")
  
  val L_out = (L - poolSize) / stride + 1
  val shift = log2Up(poolSize)
  // ACT-01: elements left unconsumed by the last window. They belong to the
  // current frame and MUST be drained before the next frame starts, otherwise
  // every following sequence is shifted by `residue` elements.
  val residue = (L - poolSize) % stride
  
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(L, channels), lanes = channels))
    val c = master(Tensor(dataType, Seq(L_out, channels), lanes = channels))
  }
  
  val shiftRegs = Seq.fill(channels)(Vec(Reg(dataType), poolSize))
  shiftRegs.foreach(_.foreach(r => r.init(r.getZero.asInstanceOf[T])))
  
  val elementCount = Counter(L)
  val windowCount = Counter(L_out)
  
  io.a.stream.ready := False
  io.c.stream.valid := False
  
  // Combinatorial average computation (Adder-Tree then shift)
  def buildAdderTree(nodes: Seq[Data]): Data = {
    if (nodes.length == 1) return nodes.head
    val nextLevel = nodes.grouped(2).map {
      case Seq(a) => a
      case Seq(a, b) =>
        (a, b) match {
          case (valA: SInt, valB: SInt) => valA + valB
          case (valA: UInt, valB: UInt) => valA + valB
          case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(valA, valB)
          case _ => throw new Exception("Type not supported in buildAdderTree")
        }
    }.toSeq
    buildAdderTree(nextLevel)
  }

  for (ch <- 0 until channels) {
    shiftRegs(ch)(0) match {
      case _: SInt => 
        val resizedNodes = shiftRegs(ch).map(_.asInstanceOf[SInt].resize(dataType.getBitsWidth + shift))
        val acc = buildAdderTree(resizedNodes).asInstanceOf[SInt]
        io.c.stream.payload(ch).assignFrom(
          RequantizeMath.shiftSaturate(acc, shift, dataType.getBitsWidth, rounding).asInstanceOf[T])
      case _: UInt =>
        val resizedNodes = shiftRegs(ch).map(_.asInstanceOf[UInt].resize(dataType.getBitsWidth + shift))
        val acc = buildAdderTree(resizedNodes).asInstanceOf[UInt]
        io.c.stream.payload(ch).assignFrom(
          RequantizeMath.shiftRound(acc, shift, dataType.getBitsWidth, rounding).asInstanceOf[T])
      case f: spinalML.dtypes.FloatML =>
        val acc = buildAdderTree(shiftRegs(ch).toSeq).asInstanceOf[spinalML.dtypes.FloatML]
        val avg = spinalML.dtypes.FloatML(f.expBits, f.mantBits)
        avg.sign := acc.sign
        avg.mantissa := acc.mantissa
        val shiftedExp = acc.exponent.intoSInt - shift
        when(shiftedExp <= 0 || acc.exponent === 0) {
          avg.exponent := 0
          avg.mantissa := 0
          avg.sign := False
        } otherwise {
          avg.exponent := shiftedExp.asUInt.resized
        }
        io.c.stream.payload(ch).assignFrom(avg.asInstanceOf[T])
      case _ => 
        throw new Exception("Data type not supported for AvgPool")
    }
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
             if (residue > 0) goto(statePurge) else goto(stateDone)
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
    
    // Drain the per-frame tail ignored by the last window (ACT-01). Number of
    // beats is an elaboration-time constant, so the state is not even built
    // when the stride divides the window span (legacy aligned shapes).
    val purgeCounter = Counter(residue max 1)
    val statePurge: State = if (residue > 0) new State {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.valid) {
          purgeCounter.increment()
          when(purgeCounter.willOverflowIfInc) {
            goto(stateDone)
          }
        }
      }
    } else null

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
  def apply[T <: Data](a: Tensor[T], poolSize: Int, stride: Int, outLanes: Int = -1,
                       rounding: RoundingMode = RoundingConfig.current): Tensor[T] = {
    require(a.shape.length == 2, "AvgPool1D expects a 2D tensor [L, channels]")
    val channels = a.shape(1)
    val in = if (a.lanes != channels) repack(a, channels) else a
    
    val comp = AvgPool1DOp(in.dataType, in.shape(0), channels, poolSize, stride, rounding)
    comp.io.a <> in
    val rawC = comp.io.c
    val finalLanes = if (outLanes > 0) outLanes else channels
    if (rawC.lanes != finalLanes) repack(rawC, finalLanes) else rawC
  }
}
