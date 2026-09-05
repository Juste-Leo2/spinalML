// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.poolings

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor

/**
 * BRAM delay-line: delays its input stream by exactly `depth` accepted beats.
 * Circular Mem + readSync guarantees Block RAM inference (guideline opsSupport.md).
 * Read address = ptr + 1 so that the popped value is aligned with the current
 * input beat even under arbitrary valid/ready stalls.
 */
case class LineBuffer2D[T <: Data](dataType: HardType[T], depth: Int) extends Component {
  require(depth >= 2, "LineBuffer2D depth (W*C) must be >= 2")

  val io = new Bundle {
    val push = slave Flow (dataType())
    val pop = master Flow (dataType())
  }

  val mem = Mem(dataType, depth)
  mem.init(Seq.fill(depth)(dataType().getZero))

  val ptr = Counter(depth)
  val rdAddr = Mux(ptr.value === depth - 1, U(0, log2Up(depth) bits), ptr.value + 1)

  mem.write(ptr.value, io.push.payload, enable = io.push.valid)
  when(io.push.valid) { ptr.increment() }

  io.pop.valid := RegNext(io.push.valid) init (False)
  io.pop.payload := mem.readSync(rdAddr, enable = io.push.valid)
}

/**
 * MaxPool2DOp: 2D Max Pooling with multi-channel support.
 * Input A: shape [H, W, C], lanes = 1 (one element per beat, row-major [H][W][C])
 * Output C: shape [H_out, W_out, C], lanes = C (one pooled pixel per beat)
 * H_out = (H - K) / stride + 1, W_out = (W - K) / stride + 1
 */
case class MaxPool2DOp[T <: Data](dataType: HardType[T], H: Int, W: Int, C: Int, K: Int, stride: Int) extends Component {
  require(H >= K && W >= K, "Image dimensions must be >= kernel size")
  require(C >= 1 && stride >= 1)

  val H_out = (H - K) / stride + 1
  val W_out = (W - K) / stride + 1
  val totalWindows = H_out * W_out
  val depth = W * C

  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(H, W, C), lanes = 1))
    val c = master(Tensor(dataType, Seq(H_out, W_out, C), lanes = C))
  }

  // Line buffers: buffer i delays its input by (i+1) full rows (depth beats each)
  val lineBuffers: Seq[LineBuffer2D[T]] = Seq.tabulate(K - 1) { i =>
    val lb = LineBuffer2D(dataType, depth)
    lb.io.push.payload := (if (i == 0) io.a.stream.payload(0) else lineBuffers(i - 1).io.pop.payload)
    lb.io.push.valid := io.a.stream.fire
    lb
  }

  // Column assembly: partial latches per beat (channel index fastest)
  val tempVecs = Vec.fill(K)(Vec(Reg(dataType), C))
  tempVecs.foreach(_.foreach(r => r.init(r.getZero.asInstanceOf[T])))

  // Sliding window registers [K rows][K*C per row], shifts left by C on every pixel completion
  val winRegs = Vec.fill(K)(Vec(Reg(dataType), K * C))
  winRegs.foreach(_.foreach(r => r.init(r.getZero.asInstanceOf[T])))

  val chCnt = Counter(C)
  val xCnt = Counter(W)
  val yCnt = Counter(H)
  val outCnt = Counter(totalWindows)

  // Combinational column vector: row r = image row y-(K-1-r), column x
  val currentPixels = Vec.fill(K)(Vec(dataType, C))
  for (r <- 0 until K) {
    for (ch <- 0 until C - 1) {
      currentPixels(r)(ch) := tempVecs(r)(ch)
    }
    if (r == K - 1) {
      currentPixels(r)(C - 1) := io.a.stream.payload(0)
    } else {
      currentPixels(r)(C - 1) := lineBuffers(K - 2 - r).io.pop.payload
    }
  }

  // Combinatorial max computation (Max-Tree), per channel, over the K*K window
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

  for (ch <- 0 until C) {
    val nodes = for (r <- 0 until K; k <- 0 until K) yield winRegs(r)(k * C + ch)
    io.c.stream.payload(ch) := buildMaxTree(nodes)
  }

  // Output grid: a window is emitted at pixel (y, x) iff it lands exactly on an
  // output position: x = K-1 + i*stride, y = K-1 + j*stride, within the image bounds.
  val colBoundary = chCnt.willOverflowIfInc
  val xAligned = (xCnt.value % stride) === (K - 1) % stride
  val yAligned = (yCnt.value % stride) === (K - 1) % stride
  val windowReady = (xCnt.value >= K - 1) && (yCnt.value >= K - 1)
  val emitWindow = colBoundary && xAligned && yAligned && windowReady

  io.a.stream.ready := False
  io.c.stream.valid := False

  val fsm = new StateMachine {

    val stateFill: State = new State with EntryPoint {
      whenIsActive {
        io.a.stream.ready := True
        when(io.a.stream.fire) {
          tempVecs(K - 1)(chCnt.value) := io.a.stream.payload(0)
          for (i <- 0 until K - 1) {
            tempVecs(K - 2 - i)(chCnt.value) := lineBuffers(i).io.pop.payload
          }

          chCnt.increment()
          when(colBoundary) {
            xCnt.increment()
            when(xCnt.willOverflowIfInc) { yCnt.increment() }

            for (r <- 0 until K) {
              for (i <- 0 until K * C - C) {
                winRegs(r)(i) := winRegs(r)(i + C)
              }
              for (ch <- 0 until C) {
                winRegs(r)(K * C - C + ch) := currentPixels(r)(ch)
              }
            }

            when(emitWindow) {
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
          outCnt.increment()
          when(outCnt.willOverflowIfInc) {
            goto(stateDone)
          } otherwise {
            goto(stateFill)
          }
        }
      }
    }

    val stateDone: State = new State {
      whenIsActive {
        chCnt.clear()
        xCnt.clear()
        yCnt.clear()
        outCnt.clear()
        goto(stateFill)
      }
    }
  }
}

object maxpool2d {
  def apply[T <: Data](a: Tensor[T], poolSize: Int, stride: Int): Tensor[T] = {
    require(a.shape.length >= 2 && a.shape.length <= 3, "MaxPool2D expects a 2D [H, W] or 3D [H, W, channels] tensor")
    val C = if (a.shape.length == 3) a.shape(2) else 1
    require(a.lanes == 1, s"MaxPool2D input must have lanes = 1")

    val comp = MaxPool2DOp(a.dataType, a.shape(0), a.shape(1), C, poolSize, stride)
    comp.io.a <> a
    comp.io.c
  }
}
