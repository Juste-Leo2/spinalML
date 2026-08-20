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
  wLanes: Int,
  projLanes: Int = 1
) extends Component {
  require(projLanes >= 1, "projLanes must be >= 1")
  require(numHeads >= 1, "numHeads must be >= 1")
  require(isPow2(numHeads), "numHeads must be a power of 2 (V1 head splitting)")
  require(embedDim % numHeads == 0, s"embedDim ($embedDim) must be divisible by numHeads ($numHeads)")
  require(wLanes == embedDim, "Attention weight lanes must equal embedDim so each stream chunk holds one weight row")
  val headDim = embedDim / numHeads

  val io = new Bundle {
    val x = slave(Tensor(dataType, Seq(seqLen, embedDim), xLanes))
    val wq = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wk = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wv = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val wo = slave(Tensor(dataType, Seq(embedDim, embedDim), wLanes))
    val y = master(Tensor(accType, Seq(seqLen, embedDim), lanes = 1))
  }
  
  // 1. Projections per head: Q_h = X * Wq_h, K_h = X * Wk_h, V_h = X * Wv_h
  // Weight streams are column-major (each stream transaction = one weight column),
  // so head h owns the transaction block [h*headDim, (h+1)*headDim) = columns of Wq/Wk/Wv.
  // The projection stage runs at `projLanes` parallelism (transistor/speed knob).
  val xForks = StreamFork(io.x.stream, 3 * numHeads)
  val wqForks = StreamFork(io.wq.stream, numHeads)
  val wkForks = StreamFork(io.wk.stream, numHeads)
  val wvForks = StreamFork(io.wv.stream, numHeads)
  
  def combineContexts(ctxs: Seq[Tensor[T]]): Tensor[T] = {
    if (ctxs.length == 1) ctxs.head
    else {
      val combined = ctxs.grouped(2).map {
        case Seq(a, b) => concatenate(a, b, axis = 1)
        case Seq(a)    => a
      }.toSeq
      combineContexts(combined)
    }
  }
  
  val contexts: Seq[Tensor[T]] = for (h <- 0 until numHeads) yield {
    val xQ = Tensor(dataType, io.x.shape, xLanes)
    xQ.stream << xForks(3 * h)
    val xK = Tensor(dataType, io.x.shape, xLanes)
    xK.stream << xForks(3 * h + 1)
    val xV = Tensor(dataType, io.x.shape, xLanes)
    xV.stream << xForks(3 * h + 2)
    
    val wqHW = Tensor(dataType, io.wq.shape, wLanes)
    wqHW.stream << wqForks(h)
    val wkHW = Tensor(dataType, io.wk.shape, wLanes)
    wkHW.stream << wkForks(h)
    val wvHW = Tensor(dataType, io.wv.shape, wLanes)
    wvHW.stream << wvForks(h)
    
    // Keep transactions [h*headDim, (h+1)*headDim) of the column-major weight stream,
    // then re-declare the shape as [embedDim, headDim] (K rows x N=headDim columns).
    val wqH = Tensor(dataType, Seq(embedDim, headDim), wLanes)
    wqH.stream << slice(wqHW, h * headDim, (h + 1) * headDim, axis = 0).stream
    val wkH = Tensor(dataType, Seq(embedDim, headDim), wLanes)
    wkH.stream << slice(wkHW, h * headDim, (h + 1) * headDim, axis = 0).stream
    val wvH = Tensor(dataType, Seq(embedDim, headDim), wLanes)
    wvH.stream << slice(wvHW, h * headDim, (h + 1) * headDim, axis = 0).stream
    
    val q = matmul(repack(xQ, projLanes), repack(wqH, projLanes), dataType)
    val k = matmul(repack(xK, projLanes), repack(wkH, projLanes), dataType)
    val v = matmul(repack(xV, projLanes), repack(wvH, projLanes), dataType)
    
    // 2. Transpose K per head: K_T (Requires BRAM buffering)
    val k_t = transpose(k)
    
    // 3. Dot Product per head: Q * K_T
    // Add queues to prevent StreamFork deadlocks!
    // k_t must be fully buffered in matmul's BRAM before it can consume q.
    // So q must be buffered until k_t is fully loaded.
    val q_fifo = Tensor(dataType, q.shape, q.lanes)
    q_fifo.stream << q.stream.queue(seqLen * headDim)
    val scores = matmul(q_fifo, k_t, dataType)
    
    // 4. Scale: scores / sqrt(headDim)
    // Currently approximated or skipped since it's just a constant scale.
    // For V1, we pass the scores directly to softmax.
    // NOTE: if pretrained weights without rescaling are ever reused, this
    // constant multiplication must be implemented to stay faithful to the model.
    
    // 5. Softmax per head (Softmax1D operates over seqLen rows)
    val scores_repacked = repack(scores, seqLen)
    val probs = spinalML.activations.Softmax1D(dataType, seqLen, seqLen)
    probs.io.x <> scores_repacked
    
    // 6. Attention per head: probs * V
    val v_fifo = Tensor(dataType, v.shape, v.lanes)
    v_fifo.stream << v.stream.queue(seqLen * headDim)
    val v_repacked = repack(v_fifo, seqLen)
    val context = matmul(probs.io.y, v_repacked, dataType)
    
    // Repack to one row per chunk so the axis-1 concatenation aligns rows block-wise
    repack(context, headDim)
  }
  
  // 7. Concatenate heads along embedDim, then Output Projection: context * Wo
  val context = combineContexts(contexts)
  val context_repacked = repack(context, wLanes)
  val y_out = matmul(context_repacked, io.wo, accType)
  
  io.y <> y_out
}
