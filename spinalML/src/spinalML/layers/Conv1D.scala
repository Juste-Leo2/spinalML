// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._

/**
 * Conv1DLayer: A 1D Convolutional Layer (Single Input/Output Channel).
 * Formula: Y = Conv1D(X, W) + b
 */
case class Conv1DLayer[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  L_in: Int,
  inChannels: Int,
  outChannels: Int,
  K: Int,
  outLanes: Int,
  tileSize: Int = 1024,
  parallelN: Boolean = false,
  temporal: Int = 0,
  inLanes: Int = 1,
  convOutLanes: Int = 1,
  // K-pass slice engine over the flattened K*inChannels axis. Same
  // contract as Conv2DLayer: pass 0 seeds zeros, passes > 0 seed from
  // spillIn, non-final passes drain M*N partials to spillOut, the final
  // pass drives io.y with the real bias once. The A-window sits on the
  // seq2col cols stream (beats per cols row); the pass loop (re-fire,
  // slice addresses, pass counting) lives in Sequential.
  spill: Boolean = false,
  spillKSlice: Int = -1,
  // Trailing pad elements closing the seed region's last AXI beat (see
  // MatmulOp spillPadElems). Computed by Sequential from the region beats;
  // 0 = beat-exact region.
  spillPadElems: Int = 0
) extends Component {
  val L_out = L_in - K + 1
  val windowSize = K * inChannels
  require(!spill || temporal >= 1,
    s"Conv1DLayer spill=true requires temporal >= 1 (the spill drain reuses the windowed row drain)")
  require(!spill || (spillKSlice > 0 && windowSize % spillKSlice == 0),
    s"Conv1DLayer spillKSlice=$spillKSlice must be a positive divisor of windowSize=$windowSize")
  require(!spill || spillKSlice % outLanes == 0,
    s"Conv1DLayer spillKSlice=$spillKSlice must be a multiple of outLanes=$outLanes " +
      "(pass-internal chunking must match the replica fadd order exactly)")
  require(!spill || !parallelN,
    s"Conv1DLayer spill=true requires the sequential-N matmul (parallelN=false)")

  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(L_in, inChannels), lanes = inLanes))
    val w = slave(Tensor(dataType, Seq(K * inChannels, outChannels), lanes = outLanes))
    val b = slave(Tensor(accType, Seq(1, outChannels), lanes = 1))
    val y = master(Tensor(accType, Seq(L_out, outChannels), lanes = convOutLanes))
    // Command-boundary re-arm for the internal weight buffer (see MatmulOp)
    val reArm = in Bool()
    // Command-boundary re-arm for the bias cache (spill=true only: the
    // bias-zero mux re-arms per pass; non-spill keeps bias_add on io.reArm
    // as before, so legacy instantiations are untouched).
    val biasReArm = if (spill) Some(in Bool()) else None
    // Spill ports (spill=true only). Same level contract as MatmulOp:
    // the Sequential pass controller holds passFirst/passLast/passIdx stable
    // per pass (single source of truth — never counted locally).
    val spillIn = if (spill) Some(slave(Tensor(accType, Seq(L_out, outChannels), lanes = 1))) else None
    val spillOut = if (spill) Some(master(Tensor(accType, Seq(L_out, outChannels), lanes = 1))) else None
    val passFirst = if (spill) Some(in Bool()) else None
    val passLast = if (spill) Some(in Bool()) else None
    val passDone = if (spill) Some(out Bool()) else None
    val passIdx = if (spill) Some(in UInt((log2Up(windowSize / spillKSlice) max 1) bits)) else None
  }

  // Seq2Col windows ([L_out, K*inChannels], lanes = outLanes),
  // self-restarting per frame (stateDone clears counters): every A
  // re-stream reproduces the identical cols sweep, so no window state
  // ever leaks across passes.
  val cols = seq2col(io.x, K, outLanes)

  // A-window on the cols stream (mirror of Conv2DLayer): pass p
  // consumes beats [p*Ks,(p+1)*Ks) of each cols row (beat-aligned by
  // Ks%outLanes==0). Non-window beats are accepted-and-dropped so upstream
  // (seq2col) never stalls; passIdx is prelude-stable during flow.
  // SLICE GEOMETRY: the engine is shaped [M,Ks]x[Ks,N] per pass.
  val rowBeatsA = if (spill) windowSize / outLanes else 1
  val winBeatsA = if (spill) spillKSlice / outLanes else 1
  val aBeatInRow = if (spill) Some(Reg(UInt((log2Up(rowBeatsA) max 1) bits)) init(0)) else None
  val colsWin = if (spill) {
    val winLo = (io.passIdx.get * U(winBeatsA, 16 bits)).resize(16 bits)
    val inWin = aBeatInRow.get.resize(16 bits) >= winLo &&
      aBeatInRow.get.resize(16 bits) < winLo + U(winBeatsA, 16 bits)
    val gated = Tensor(dataType, cols.shape, cols.lanes)
    gated.stream.valid := cols.stream.valid && inWin
    gated.stream.payload := cols.stream.payload
    cols.stream.ready := !inWin || gated.stream.ready
    when(cols.stream.fire) {
      when(aBeatInRow.get === U(rowBeatsA - 1, (log2Up(rowBeatsA) max 1) bits)) {
        aBeatInRow.get := 0
      } otherwise {
        aBeatInRow.get := aBeatInRow.get + 1
      }
    }
    gated
  } else cols
  // Slice views (stream aliases): windowed cols [M,Ks],
  // fetched W slice [Ks,N]. Beat counts match the flows exactly.
  val matmulA = if (spill) {
    val s = Tensor(dataType, Seq(L_out, spillKSlice), outLanes)
    s.stream << colsWin.stream
    s
  } else colsWin
  val matmulW = if (spill) {
    val s = Tensor(dataType, Seq(spillKSlice, outChannels), outLanes)
    s.stream << io.w.stream
    s
  } else io.w

  // Matmul cols*W ([L_out, K*inChannels] x [K*inChannels, outChannels]
  // -> [L_out, outChannels]; reArm re-arms the internal B buffer,
  // temporal bounds rows in flight).
  val matmulResult = matmul(matmulA, matmulW, accType, parallelN = parallelN, reArm = Some(io.reArm), temporal = temporal,
    spill = spill, passFirst = io.passFirst, passLast = io.passLast,
    spillSource = io.spillIn, spillSink = io.spillOut, passDone = io.passDone, spillPadElems = spillPadElems)

  // Add bias via the bias-zero mux (mirror of Conv2DLayer): BiasAddOp always
  // consumes exactly N beats per pass, but on non-final passes the beats come
  // from an on-chip zero source; the real bias is added once, final pass only.
  val bForAdd: Tensor[TAcc] = if (spill) {
    val zeroRemain = Reg(UInt(log2Up(outChannels + 1) bits)) init (outChannels)
    when(io.biasReArm.get) {
      zeroRemain := outChannels
    }
    val bMux = Tensor(accType, Seq(1, outChannels), lanes = 1)
    when(io.passLast.get) {
      bMux.stream.valid := io.b.stream.valid
      bMux.stream.payload := io.b.stream.payload
      io.b.stream.ready := bMux.stream.ready
    } otherwise {
      // reArm/fire race guard (same family as the LinearLayer/BiasAddOp
      // guards): bubble the zero offer on the reArm cycle so the reset and
      // the decrement can never collide on one beat.
      bMux.stream.valid := zeroRemain =/= 0 && !io.biasReArm.get
      bMux.stream.payload(0).assignFromBits(B(0, widthOf(accType) bits))
      io.b.stream.ready := False
      when(bMux.stream.fire) {
        zeroRemain := zeroRemain - 1
      }
    }
    bMux
  } else {
    io.b
  }
  val biased = bias_add(matmulResult, bForAdd, reArm = if (spill) Some(io.biasReArm.get) else Some(io.reArm))
  if (convOutLanes == biased.lanes) io.y <> biased else io.y <> repack(biased, convOutLanes)
}

object Conv1D {
  def apply[T <: Data, TAcc <: Data](
    x: Tensor[T],
    w: Tensor[T],
    b: Tensor[TAcc],
    accType: HardType[TAcc],
    parallelN: Boolean = false,
    reArm: Option[Bool] = None,
    temporal: Int = 0,
    outLanes: Int = 1
  ): Tensor[TAcc] = {
    val inChannels = if (x.shape.length == 2) x.shape(1) else 1
    val outChannels = w.shape(1)
    val K = w.shape(0) / inChannels

    val comp = Conv1DLayer(
      x.dataType, accType, x.shape(0), inChannels, outChannels, K,
      outLanes = w.lanes, tileSize = w.shape(0), parallelN = parallelN, temporal = temporal,
      inLanes = x.lanes, convOutLanes = outLanes
    )
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.x <> x
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
  
  def apply[T <: Data, TAcc <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    apply(x, w, b, accType, false, None)
  }

  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T], parallelN: Boolean): Tensor[T] = {
    apply(x, w, b, x.dataType, parallelN, None)
  }

  def apply[T <: Data](x: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(x, w, b, x.dataType, false, None)
  }
}
