package spinalML.attention

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.ops._
import spinalML.activations._

/**
 * ClassicalAttention: Scaled Dot-Product Attention Layer
 * Formula: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * Output = Attention * Wo
 */
case class ClassicalAttention(
  embedDim: Int,
  numHeads: Int = 1,
  customType: Option[HardType[Data]] = None,
  customWeightType: Option[HardType[Data]] = None
) extends AttentionCore {
  override def outType(default: HardType[Data]) = customType.getOrElse(default)
  override def weightType(default: HardType[Data]) = customWeightType.getOrElse(default)
  
  override def getOutShape(inShape: Seq[Int]): Seq[Int] = {
    require(inShape.length >= 2, "Attention requires at least 2D input shape (L, EmbedDim)")
    Seq(inShape(0), embedDim)
  }
  
  // Weights: Wq, Wk, Wv, Wo. Each is [embedDim, embedDim].
  // They are stored sequentially in memory, so total shape is [4 * embedDim, embedDim]
  override def getWeightShape(): Seq[Int] = Seq(embedDim * 4, embedDim)
  override def getBiasShape(): Seq[Int] = Seq(0) // No bias for simplicity in V1
}

case class ClassicalAttentionHW[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  seqLen: Int,
  embedDim: Int,
  numHeads: Int,
  xLanes: Int,
  wLanes: Int
) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(accType, Seq(seqLen, embedDim), lanes = 1))
  }
  
  // 1. Projections: Q = X * Wq, K = X * Wk, V = X * Wv
  // Fork X into 3 streams to compute Q, K, V in parallel
  val xForks = StreamFork(io.x.stream, 3)
  val xQ = Tensor(dataType, io.x.shape, xLanes)
  val xK = Tensor(dataType, io.x.shape, xLanes)
  val xV = Tensor(dataType, io.x.shape, xLanes)
  xQ.stream << xForks(0)
  xK.stream << xForks(1)
  xV.stream << xForks(2)

  val q = matmul(xQ, io.wq, dataType)
  val k = matmul(xK, io.wk, dataType)
  val v = matmul(xV, io.wv, dataType)

  // 2. Transpose K: K_T (Requires BRAM buffering)
  val k_t = transpose(k)
  
  // 3. Dot Product: Q * K_T
  // Add queues to prevent StreamFork deadlocks!
  // k_t must be fully buffered in matmul's BRAM before it can consume q.
  // So q must be buffered until k_t is fully loaded.
  val q_fifo = Tensor(dataType, q.shape, q.lanes)
  q_fifo.stream << q.stream.queue(seqLen * embedDim)
  val scores = matmul(q_fifo, k_t, dataType)
  
  // 4. Scale: scores / sqrt(embedDim)
  // Currently approximated or skipped since it's just a constant scale
  // For V1, we pass the scores directly to softmax
  
  // 5. Softmax (Requires Softmax1D to operate over seqLen)
  // Softmax operates on the last dimension by default.
  // Wait, scores is [seqLen, seqLen]. Softmax over dim=1 (the rows).
  val scores_repacked = repack(scores, seqLen)
  val probs = spinalML.activations.Softmax1D(dataType, seqLen, seqLen)
  probs.io.x <> scores_repacked
  
  // 6. Attention: probs * V
  val v_fifo = Tensor(dataType, v.shape, v.lanes)
  v_fifo.stream << v.stream.queue(seqLen * embedDim)
  val v_repacked = repack(v_fifo, seqLen)
  val context = matmul(probs.io.y, v_repacked, dataType)
  
  // 7. Output Projection: context * Wo
  val context_repacked = repack(context, wLanes)
  val y_out = matmul(context_repacked, io.wo, accType)
  
  io.y <> y_out
}
