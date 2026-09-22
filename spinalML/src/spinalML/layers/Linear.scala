// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.layers

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._

/**
 * LinearLayer: A fully connected dense layer.
 * Formula: Y = Matmul(A, W) + b
 *
 * Weight-only quantization (wXaY): when `weightType` is an SInt format
 * (I4/I8), the weights are dequantized to the activation dtype through a
 * scaled cast before the float matmul:
 *   Y = Matmul(A, FloatML(W) * scales) + b
 * `weightScales` is either per-tensor (length 1) or per-channel
 * (length = number of weight stream beats, i.e. N columns).
 * Same-dtype weights keep the legacy direct wiring.
 */
case class LinearLayer[T <: Data, TW <: Data, TAcc <: Data](
  dataType: HardType[T],
  weightType: HardType[TW],
  accType: HardType[TAcc],
  shapeA: Seq[Int],
  shapeW: Seq[Int],
  lanes: Int,
  weightScales: Seq[Double] = Seq(1.0),
  tileSize: Int = 1024,
  parallelN: Boolean = false,
  temporal: Int = 0,
  // S1 compute-side spill (docs/ddr_final_impl.md): K-pass slice engine. The
  // layer streams one K-slice per pass (S2 re-fires); pass 0 seeds zeros,
  // passes > 0 seed from spillIn, non-final passes drain partials to
  // spillOut, the final pass drives io.y with the real bias added once.
  spill: Boolean = false,
  // Trailing pad elements closing the seed region's last AXI beat (see
  // MatmulOp spillPadElems). Computed by Sequential from the region beats;
  // 0 = beat-exact region.
  spillPadElems: Int = 0
) extends Component {
  require(!spill || temporal >= 1,
    s"LinearLayer spill=true requires temporal >= 1 (the spill drain reuses the windowed row drain)")
  val M = shapeA(0)
  val K = shapeA(1)
  val N = shapeW(1)

  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val w = slave(Tensor(weightType, shapeW, lanes))
    val b = slave(Tensor(accType, Seq(1, N), lanes = 1)) // Bias matches accumulator type
    val y = master(Tensor(accType, Seq(M, N), lanes = 1)) // Output can be processed sequentially
    // Command-boundary re-arm for the internal weight buffer (see MatmulOp)
    val reArm = in Bool()
    // Command-boundary re-arm for the bias cache (see BiasAddOp)
    val biasReArm = in Bool()
    // S1 spill ports (spill=true only). Same level contract as MatmulOp:
    // the S2 pass controller holds passFirst/passLast stable per pass.
    val spillIn = if (spill) Some(slave(Tensor(accType, Seq(M, N), lanes = 1))) else None
    val spillOut = if (spill) Some(master(Tensor(accType, Seq(M, N), lanes = 1))) else None
    val passFirst = if (spill) Some(in Bool()) else None
    val passLast = if (spill) Some(in Bool()) else None
    val passDone = if (spill) Some(out Bool()) else None
  }
  
  // Weight-only quantization path: SInt weights feeding a FloatML activation
  // domain are dequantized before the (float) matmul. Any other combination
  // keeps the legacy direct wiring (uniform dtype matmul).
  private val needsDequant = (dataType(), weightType()) match {
    case (_: FloatML, _: SInt) => true
    case _ => false
  }
  
  val wForMatmul =
    if (needsDequant) cast(io.w, dataType, weightScales)
    else io.w.asInstanceOf[Tensor[T]]
  
  // 1. Matrix Multiplication: A * W_deq (reArm re-arms the internal B buffer,
  //    which carries this layer's weights; temporal bounds the rows in flight)
  val matmulResult = matmul(io.a, wForMatmul, accType, parallelN = parallelN, reArm = Some(io.reArm), temporal = temporal,
    spill = spill, passFirst = io.passFirst, passLast = io.passLast,
    spillSource = io.spillIn, spillSink = io.spillOut, passDone = io.passDone, spillPadElems = spillPadElems)

  // 2. Add Bias (Broadcast): (A * W_deq) + b
  // S1 bias-zero mux: BiasAddOp always consumes exactly N beats per pass
  // (its FSM is untouched), but on non-final passes the beats come from an
  // on-chip zero source — no DDR bias fetch, bias added once on the final
  // pass. The zero generator offers exactly N beats per pass and re-arms
  // with the bias cache (same per-pass pulse); over-consumption would
  // backpressure visibly instead of aliasing data. Exactness of x + 0 per
  // dtype is the S1 gate (MatmulSpillTest); on failure the fallback is a
  // BiasAddOp bypass on non-final passes.
  val bForAdd: Tensor[TAcc] = if (spill) {
    val zeroRemain = Reg(UInt(log2Up(N + 1) bits)) init (N)
    when(io.biasReArm) {
      zeroRemain := N
    }
    val bMux = Tensor(accType, Seq(1, N), lanes = 1)
    when(io.passLast.get) {
      bMux.stream.valid := io.b.stream.valid
      bMux.stream.payload := io.b.stream.payload
      io.b.stream.ready := bMux.stream.ready
    } otherwise {
      bMux.stream.valid := zeroRemain =/= 0
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
  io.y <> bias_add(matmulResult, bForAdd, reArm = Some(io.biasReArm))
}

object Linear {
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc], tileSize: Int, parallelN: Boolean, reArm: Option[Bool] = None, biasReArm: Option[Bool] = None, temporal: Int = 0): Tensor[TAcc] = {
    val comp = LinearLayer(a.dataType, a.dataType, accType, a.shape, w.shape, a.lanes, Seq(1.0), tileSize, parallelN, temporal)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.biasReArm := biasReArm.getOrElse(False)
    comp.io.a <> a
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }

  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[TAcc], accType: HardType[TAcc]): Tensor[TAcc] = {
    apply(a, w, b, accType, 1024, false, None)
  }

  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T], tileSize: Int, parallelN: Boolean): Tensor[T] = {
    apply(a, w, b, a.dataType, tileSize, parallelN, None)
  }

  def apply[T <: Data](a: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(a, w, b, a.dataType, 1024, false, None)
  }

  // Weight-only quantization (wXaY): weights stored as SInt (I4/I8) plus
  // compile-time scale(s), activations in the float domain. No default
  // arguments here: only one overload of Linear may define defaults.
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], w: Tensor[SInt], b: Tensor[TAcc], accType: HardType[TAcc], weightScales: Seq[Double], parallelN: Boolean, tileSize: Int, reArm: Option[Bool], biasReArm: Option[Bool], temporal: Int): Tensor[TAcc] = {
    val comp = LinearLayer(a.dataType, w.dataType, accType, a.shape, w.shape, a.lanes, weightScales, tileSize, parallelN, temporal)
    comp.io.reArm := reArm.getOrElse(False)
    comp.io.biasReArm := biasReArm.getOrElse(False)
    comp.io.a <> a
    comp.io.w <> w
    comp.io.b <> b
    comp.io.y
  }
}
