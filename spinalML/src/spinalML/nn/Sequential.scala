// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinalML.memory._
import spinalML.tensors.Tensor
import spinalML.{Target, FpgaFamily, PdkFamily}
import spinalML.arithmetic.ArithmeticConfig
import spinalML.utils.{MemLayout, SimLog}
import spinalML.dtypes.FloatML
import spinalML.layers.{Conv1D => Conv1DHW, Conv2D => Conv2DHW, Linear => LinearHW, batchnorm}
import spinalML.ops.{reshape, repack, flatten, cast}
import spinalML.activations.{relu, leaky_relu, sigmoid, tanh}
import spinalML.poolings.{maxpool1d, avgpool1d, maxpool2d, avgpool2d}
import spinalML.attention._

case class Sequential(
  globalDataType: HardType[Data],
  inputShape: Seq[Int],
  layers: Seq[LayerSpec],
  axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 8),
  // Phase-2a weight residency: when set, exposes the run-mode control plane
  // (see Accelerator CSR map). Direct users of this component keep today's
  // always-fetch behaviour when left at the default.
  val weightResidency: Boolean = false,
  // Phase-3 activation tiling: number of IMAGE ROWS per band (vertical
  // stripe, full width). Each inference is fetched as ceil(H/tileHeight)
  // back-to-back 2D patch commands; on-chip image buffering is sized to ONE
  // band instead of the whole tensor. The consumer stream is UNCHANGED:
  // bands concatenate into the same continuous row stream (the seam is a
  // stream stall — im2col's carried state is the halo, proven in S2).
  // tileHeight <= 0 means "one band = the whole image" (legacy behaviour).
  val tileHeight: Int = -1,
  // M3: rows-in-flight bound of the matmul-based layers' output accumulator
  // tables. 0 = legacy full MxN table (the M4A8 Conv matmul is the LUT wall:
  // M=576 windows x N=2 channels in registers + the wide index mux);
  // > 0 = each row is drained as soon as it completes and the table shrinks
  // to <= min(temporal, M) x N slots. Bit-exactness is preserved by
  // construction (the fadd sum order is unchanged).
  val temporal: Int = 0,
  val inLanes: Int = 1,
  // Phase-1 DDR plumbing (docs/ddr_impl.md): hardware target carried by the
  // high-level API so the same modelSpec elaborates for sim / FPGA / ASIC.
  // Default Simulation keeps CI behavior bit-identical (layers still use
  // DspConfig.default = Target.current until Phase 4 threads this through).
  val target: Target = Target.Simulation,
  // S0 compute-side spill (docs/ddr_final_impl.md): on-chip replay budget in
  // bytes for a spilling layer whose A operand is NOT DDR-backed (deep
  // layer): pass 0 snoops the A stream into a replay Mem, passes > 0 re-read
  // it. Layers above budget (and not on node 0) fail elaboration with a
  // pointer at A-spill (v2). 0 disables the replay path (node-0 spill only).
  val spillReplayBudgetBytes: Int = 4096
) extends Component {
  require(temporal >= 0, s"Sequential temporal=$temporal must be >= 0")
  require(spillReplayBudgetBytes >= 0,
    s"Sequential spillReplayBudgetBytes=$spillReplayBudgetBytes must be >= 0")

  // S2a/S2b compute-side spill (docs/ddr_final_impl.md): v1 supports at most
  // ONE spilling layer (a single SpillPassController in S2b; multi-spill
  // generalizes on this template later). Any SpillableGEMM layer (Linear,
  // Conv2D, ...) may spill; v1 supports at most one spilling layer (pinned
  // below). Placed before `io`: the spill write port below depends on it.
  // The spill WRITE path cannot join the shared read arbiter
  // (io.axiMaster is read-only): the drain DMAWriter gets its own write
  // master port, merged with the output dmaWriter in Accelerator (S2b 2:1
  // write arbiter). The seed reader joins allAxiMasters normally.
  val spillLayerIdx: Option[Int] = {
    val idx = layers.zipWithIndex.collect { case (l: SpillableGEMM, i) if l.spilling => i }
    require(idx.size <= 1,
      s"Sequential: ${idx.size} spilling layers (${idx.mkString(",")}) — v1 supports at most one " +
        "(single pass controller, see docs/ddr_final_impl.md S2)")
    idx.headOption
  }
  // Leaf ID width for the spill write path (one route bit for the S2b 2:1
  // write arbiter in Accelerator — mirror of the read-side routeBits below).
  val spillWriteLeafCfg = axiConfig.copy(idWidth = axiConfig.idWidth - 1)
  if (spillLayerIdx.nonEmpty)
    require(axiConfig.idWidth >= 1,
      s"Sequential: axiConfig.idWidth (${axiConfig.idWidth}) leaves no route bit for the spill write arbiter")

  /** Arithmetic policy derived from the high-level target (Phase 4 will
    * thread this into the layer instantiations below). */
  val arithmeticConfig: ArithmeticConfig = ArithmeticConfig(target = target)

  // ============================================================
  // 0. Topology analysis (pure elaboration-time, no hardware yet)
  // ============================================================
  // Node 0 is the network input; node k (k >= 1) is the output of layers(k - 1).
  // Standard layers implicitly consume the previous node (index = their position);
  // merge layers (Add / Concat) consume two explicitly referenced earlier nodes,
  // which keeps the graph acyclic by construction.
  val nNodes = layers.length + 1

  private def consumedNodes(layerIdx: Int): Seq[Int] = layers(layerIdx) match {
    case ad: Add    => Seq(ad.a, ad.b)
    case cc: Concat => Seq(cc.a, cc.b)
    case _          => Seq(layerIdx)
  }

  private def sameDtype(x: HardType[Data], y: HardType[Data]): Boolean = {
    val dx = x(); val dy = y()
    dx.getClass == dy.getClass && dx.getBitsWidth == dy.getBitsWidth
  }

  private def hardTypeName(t: HardType[Data]): String = {
    val d = t()
    s"${d.getClass.getSimpleName}(${d.getBitsWidth}b)"
  }

  val consumers = Array.fill(nNodes)(scala.collection.mutable.ArrayBuffer[Int]())
  for (i <- layers.indices; n <- consumedNodes(i)) {
    require(n <= i,
      s"Layer $i (${layers(i)}) references node $n: DAG references must point to earlier nodes [0..$i]")
    consumers(n) += i
  }
  consumers(nNodes - 1) += -1 // -1 marks the accelerator output

  for (n <- 0 until nNodes) {
    require(consumers(n).nonEmpty,
      s"Node $n (${if (n == 0) "network input" else s"output of ${layers(n - 1)}"}) is never consumed")
  }

  val nodeShapes = scala.collection.mutable.ArrayBuffer[Seq[Int]](inputShape)
  val nodeTypes = scala.collection.mutable.ArrayBuffer[HardType[Data]](globalDataType)
  val nodeLanes = scala.collection.mutable.ArrayBuffer[Int](inLanes)

  // SimLog audit record: per-layer weight footprint, filled by the build loop.
  private val auditWeightLanes = scala.collection.mutable.ArrayBuffer[Int]()
  private val auditWeightElements = scala.collection.mutable.ArrayBuffer[Int]()
  private val auditWeightBeats = scala.collection.mutable.ArrayBuffer[Int]()

  for (i <- layers.indices) {
    val l = layers(i)
    val outShape = l match {
      case ad: Add =>
        val sa = nodeShapes(ad.a); val sb = nodeShapes(ad.b)
        require(sa == sb,
          s"Add(${ad.a}, ${ad.b}) requires identical shapes: $sa vs $sb")
        sa
      case cc: Concat =>
        val sa = nodeShapes(cc.a); val sb = nodeShapes(cc.b)
        require(sa.length == sb.length && sa.tail == sb.tail,
          s"Concat(${cc.a}, ${cc.b}, axis=0) requires identical tail shapes: $sa vs $sb")
        sa.updated(0, sa.head + sb.head)
      case _ => l.getOutShape(nodeShapes(i))
    }
    val outType = l match {
      case ad: Add =>
        require(sameDtype(nodeTypes(ad.a), nodeTypes(ad.b)),
          s"Add(${ad.a}, ${ad.b}) requires identical dtypes: ${nodeTypes(ad.a)} vs ${nodeTypes(ad.b)} (insert a Cast layer on one branch)")
        nodeTypes(ad.a)
      case cc: Concat =>
        require(sameDtype(nodeTypes(cc.a), nodeTypes(cc.b)),
          s"Concat(${cc.a}, ${cc.b}) requires identical dtypes: ${nodeTypes(cc.a)} vs ${nodeTypes(cc.b)} (insert a Cast layer on one branch)")
        nodeTypes(cc.a)
      case _ => l.outType(nodeTypes(i))
    }
    val outLane = l match {
      case c: Conv2D => c.lanes
      case l: Linear => l.lanes
      case c: Conv1D => c.lanes
      case mp: MaxPool1D => if (mp.lanes > 0) mp.lanes else (if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1)
      case ap: AvgPool1D => if (ap.lanes > 0) ap.lanes else (if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1)
      case mp2: MaxPool2D => mp2.lanes
      case ap2: AvgPool2D => ap2.lanes
      case bn: BatchNorm1D => if (bn.lanes > 0) bn.lanes else bn.features
      case ln: LayerNorm1D => if (ln.lanes > 0) ln.lanes else ln.features
      case a: ClassicalAttention => a.lanes
      case sm: Softmax => sm.lanes
      case rp: Repack => rp.newLanes
      case ad: Add => nodeLanes(ad.a)
      case cc: Concat => cc.lanes
      case _ => nodeLanes(i)
    }
    nodeShapes += outShape
    nodeTypes += outType
    nodeLanes += outLane
  }

  def computeFinalShape(): (Seq[Int], HardType[Data], Int) = (nodeShapes.last, nodeTypes.last, nodeLanes.last)
  val (finalShape, finalType, finalLanes) = computeFinalShape()

  val io = new Bundle {
    val start = slave(Event)
    val imgBaseAddress = in UInt(axiConfig.addressWidth bits)
    val weightsBaseAddress = in UInt(axiConfig.addressWidth bits)
    // S2a compute-side spill (docs/ddr_final_impl.md): DDR base of the
    // accumulator spill region (M*N full-width partials per spilling layer).
    // Wired by Accelerator to CSR 0x34 + runtime cursor; consumed by the S2b
    // spill DMA pair (read seed / write back). Undriven-safe in S2a: no logic
    // reads it yet, so direct-Sequential sims are unaffected.
    val spillBaseAddress = in UInt(axiConfig.addressWidth bits)
    // S2b spill drain path: the seed reader joins the shared read arbiter,
    // but this write master carries the drain DMAWriter (io.axiMaster is
    // read-only). Some(...) only on spilling models; Accelerator merges it
    // with the output dmaWriter through a 2:1 write arbiter.
    val spillWrite = if (spillLayerIdx.nonEmpty) Some(master(Axi4WriteOnly(spillWriteLeafCfg))) else None

    // Weight-residency run-mode inputs (instantiated only when the ctor flag
    // is set; Accelerator maps them to CSR 0x10 bit0/bit1 and the 0x14 RELOAD shot)
    val weightResident = if (weightResidency) Some(in Bool()) else None
    val weightReload   = if (weightResidency) Some(in Bool()) else None
    val weightPrefetch = if (weightResidency) Some(in Bool()) else None

    val axiMaster = master(Axi4ReadOnly(axiConfig))
    val outStream = master(Tensor(finalType, finalShape, lanes = finalLanes))

    // Runtime dequantization scale input for Cast layers with runtimeScale = true
    val dequantScale = if (layers.exists { case c: Cast => c.runtimeScale; case _ => false }) Some(in Bits(32 bits)) else None

    // ---- Continuous-run frame signals (Phase 3, S1) ----------------------
    // One inference = one output frame = exactly `finalShape.product` beats.
    // These two outputs let the Accelerator auto-advance RUN mode safely:
    //   busy — high from the first START accepted until the last output beat.
    //   done — one-cycle pulse the moment a full output frame completes.
    // Counted on the STREAM HANDSHAKE (fire), so downstream backpressure
    // (ready=0) can never skip or duplicate a frame boundary.
    val busy = out Bool()
    val done = out Bool()
  }

  // ---- Weight residency control plane (Phase 2a) -------------------------
  // MODE STREAM_PER_PASS (resident low) reproduces today's wiring verbatim.
  // MODE WEIGHT_RESIDENT: weight/bias fork branches whose region is already
  // on chip accept their START beat immediately WITHOUT issuing a DMA
  // command (collective fork completion is unaffected — the image branch,
  // which always fetches, dominates its latency). Real fetches happen on
  // first use, for one pass after a RELOAD pulse, AND on the RISING EDGE of
  // the resident mode itself: at that moment the compute pointer usually
  // sits on a flipped-empty bank left over from a legacy pass, so tileReady
  // would stay low forever — one guaranteed fill makes the resident state
  // self-consistent without any host ritual. Buffers are then held with
  // `residentHold` and every later pass re-diffuses the resident bank.
  private val residentMode = io.weightResident.getOrElse(False)
  private val residentPrev = RegInit(False) init (False)
  residentPrev := residentMode
  private val residentRise = residentMode && !residentPrev
  private val weightReloadLatches = scala.collection.mutable.ArrayBuffer[Bool]()


  // --- 1. Memory Offset Calculation & DMA Instantiation ---
  val allAxiMasters = scala.collection.mutable.ArrayBuffer[Axi4ReadOnly]()
  var currentMemoryOffset = 0
  // S0 compute-side spill: exact M*N full-width partials footprint per
  // spilling layer (DDR region sized by the S2 cursor; fit-checked by
  // Accelerator.reportFit). Beat-aligned per region, MemLayout conventions.
  var spillBytesAcc = 0
  // S2a: per-pass index register of the spilling layer's W-slice fetch
  // (stays 0 in S2a — no controller yet; S2b advances it once per pass).
  // Keyed by layer index; empty when the model does not spill.
  val spillPassIdxOf = scala.collection.mutable.Map[Int, UInt]()
  // S2a: elaboration-time slice geometry of the spilling layer's fetch
  // (layer index -> (sliceElems, sliceBeats)), exposed for S2b + tests.
  val spillSliceInfo = scala.collection.mutable.Map[Int, (Int, Int)]()
  // S2d e2e observability (debug, zero behavior change): the A-side
  // K-window position (window low beat + intra-row beat counter) per
  // spilling layer, sampled by spill benches to reconstruct exactly which
  // stream beats each pass consumed.
  val spillWinLoOf = scala.collection.mutable.Map[Int, UInt]()
  val spillABeatOf = scala.collection.mutable.Map[Int, UInt]()
  // S2c: single shared pass controller (v1 = one spilling layer, pinned by
  // spillLayerIdx above), hoisted before the fetch plane: the W-slice
  // refetch (S2b, per-iteration fetch site below) AND the image re-fire
  // (S2c, image plane further below) must both see its pulses.
  // (passes, spill AXI beats for the M*N region command). M = GEMM rows:
  // Linear input rows, Conv2D total windows (H-K+1)*(W-K+1).
  val spillSpec: Option[(Int, Int)] = spillLayerIdx.map { idx =>
    val (passes, mRows, n, accType) = layers(idx) match {
      case l: Linear =>
        (l.spillPasses, nodeShapes(idx).dropRight(1).product, l.outFeatures, l.outType(nodeTypes(idx)))
      case c: Conv2D =>
        val h = nodeShapes(idx)(0)
        val w = nodeShapes(idx)(1)
        val m = (h - c.kernelSize + 1) * (w - c.kernelSize + 1)
        (c.spillPasses, m, c.outChannels, c.outType(nodeTypes(idx)))
      case other =>
        throw new IllegalArgumentException(
          s"Sequential: layer $idx ($other) spills but is not a SpillableGEMM — spillLayerIdx only collects SpillableGEMM")
    }
    val elems = mRows * n
    val accLanes = axiConfig.dataWidth / accType.getBitsWidth
    require(accLanes >= 1,
      s"Spill accumulator dtype (${accType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported spill element size")
    (passes, (elems + accLanes - 1) / accLanes)
  }
  val spillCtrl = spillSpec.map { case (p, beats) =>
    SpillPassController(p, axiConfig.addressWidth, spillBeats = beats)
  }
  // S2c A re-stream routing: an EXCLUSIVE node-0 spill re-fires the image
  // sweep from DDR (free); a shared node 0 or a deep node replays an
  // on-chip StreamTap (budget-checked in the sizing block below).
  // Re-firing a shared node would push duplicate beats into the OTHER fork
  // branch's exact-capacity TapBuffer (overflow stall or silent aliasing).
  val spillImgRefire = spillLayerIdx.exists(idx => idx == 0 && consumers(0).size == 1)
  val spillRestartA: Bool = if (spillImgRefire) spillCtrl.get.io.restartA else False

  // Every weight/bias region must start on an AXI-beat boundary: DDR
  // controllers and memory models serve bursts from the beat-aligned address,
  // so an unaligned region start would silently read the tail of the previous
  // region's word instead of the intended first elements.
  val beatBytes = axiConfig.dataWidth / 8
  def alignToBeat(offset: Int): Int = MemLayout.alignToBeat(offset, beatBytes)

  // Start triggers fork
  // Image + each weight + each bias
  val totalDmaTriggers = 1 + layers.map(l => (if(l.getWeightShape().head > 0) 1 else 0) + (if(l.getBiasShape().head > 0) 1 else 0)).sum
  val startTriggers = StreamFork(io.start, totalDmaTriggers)
  var triggerIdx = 0

  // Base AXI config for leaf DMAs (accounting for arbiter routing bits).
  // Single-stage arbiter up to arbFanIn masters, two-stage tree beyond:
  // routing budget = log2Up(fanIn) + log2Up(numGroups), always covering
  // log2Up(totalDmaTriggers). Leaf idWidth 0 is legal (Spinal requires >= 0).
  val arbFanIn = 8
  // S2b: the spill DMA pair (seed reader + drain writer) joins the read
  // arbiter but consumes no START trigger (the pass controller commands it
  // per pass) — size the routing for triggers + spill masters, fork the
  // triggers alone. Legacy models: spillDmaMasters = 0, sizing unchanged.
  // Only the seed READER sits on the read arbiter (count 1); the drain
  // writer leaves through the dedicated io.spillWrite port above.
  val spillDmaMasters = if (spillLayerIdx.nonEmpty) 1 else 0
  val totalMasters = totalDmaTriggers + spillDmaMasters
  val arbGroups = if (totalMasters <= 1) 1 else (totalMasters + arbFanIn - 1) / arbFanIn
  val routeBits = if (totalMasters <= 1) 0
    else if (totalMasters <= arbFanIn) log2Up(totalMasters)
    else log2Up(arbFanIn) + log2Up(arbGroups)
  require(axiConfig.idWidth >= routeBits,
    s"Sequential: axiConfig.idWidth (${axiConfig.idWidth}) is insufficient to arbitrate " +
    s"$totalMasters DMA masters (requires at least $routeBits bits, i.e. axiConfig.idWidth >= $routeBits).")

  val dmaAxiConfig =
    if (totalDmaTriggers == 1) axiConfig
    else axiConfig.copy(idWidth = axiConfig.idWidth - routeBits)

  // 1.1. Image DMA — banded 2D fetch (Phase-3 tiling)
  // Each band is one 2D patch command (patchHeight = band rows, baseAddress =
  // imgBase + band * bandBytes); consecutive commands are issued by the tiny
  // sequencer below on cmd.fire (the 2D reader accepts a command once the
  // previous patch's output fully crossed the trim stage — i.e. exactly when
  // the current band has landed in a bank).
  val inputDataType = globalDataType
  val dmaImg = DMAReader2D(inputDataType, inputShape, outLanes = inLanes, dmaAxiConfig)

  val elemsPerRowImg = inputShape.product / inputShape.head
  val bandRows = if (tileHeight > 0) inputShape.head.min(tileHeight) else inputShape.head
  val nBands = (inputShape.head + bandRows - 1) / bandRows
  val bandElements = bandRows * elemsPerRowImg
  val pixelBytes = inputDataType.getBitsWidth / 8
  require(pixelBytes >= 1, "image dtype narrower than 8 bits is unsupported")
  require(bandElements > 0, "band must hold at least one pixel row")
  val strideBytesImg = inputShape(1) * pixelBytes
  val bandBytes = bandRows * strideBytesImg

  // Beat-width element sanity (kept from the legacy site).
  require(axiConfig.dataWidth / inputDataType.getBitsWidth >= 1,
    s"Input dtype (${inputDataType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported image element size")

  val bandIdxW = log2Up(nBands) max 1
  val imgBandIdx = Reg(UInt(bandIdxW bits)) init(0)
  val imgBandActive = RegInit(False)
  dmaImg.io.cmd.valid := imgBandActive
  dmaImg.io.cmd.baseAddress := (io.imgBaseAddress + (U(bandBytes, axiConfig.addressWidth bits) * imgBandIdx)).resize(axiConfig.addressWidth bits)
  dmaImg.io.cmd.stride := U(strideBytesImg, axiConfig.addressWidth bits)
  dmaImg.io.cmd.patchWidth := 0 // ignored by the hardware, see DMAReader2D docs
  val imgLastBandRows = inputShape.head - (nBands - 1) * bandRows
  dmaImg.io.cmd.patchHeight := Mux(imgBandIdx === U(nBands - 1, bandIdxW bits),
    U(imgLastBandRows, 16 bits), U(bandRows, 16 bits))

  // S2c: per-pass A re-stream for an exclusive node-0 spill (DDR-backed).
  // The controller pulses restartA exactly once per pass p >= 1 (suppressed
  // under residency with refetchW); the band sequencer re-runs the full
  // patch sweep and both buffers re-arm. Safe: the prelude runs after
  // passDone + spill fence, so pass p-1's image is fully consumed — no live
  // state is cleared. Legacy models: spillRestartA is a tied False.
  when(startTriggers(triggerIdx).valid || spillRestartA) {
    imgBandActive := True
    imgBandIdx := 0
  }
  startTriggers(triggerIdx).ready := True // bander accepts the START immediately
  when(dmaImg.io.cmd.fire) {
    when(imgBandIdx === U(nBands - 1, bandIdxW bits)) {
      imgBandActive := False
    } otherwise {
      imgBandIdx := imgBandIdx + 1
    }
  }

  allAxiMasters += dmaImg.io.axiMaster
  triggerIdx += 1

  // Use Automatic Double Buffering instead of a simple queue
  // The double-buffer contract tiles exactly `depth` elements per bank: the
  // buffer MUST be sized to the exact tile size (any floor like max(16, n)
  // deadlocks small tensors, as tileReady would never assert). With phase-3
  // banding the "tile" is one band; with tileHeight <= 0 it is the whole
  // tensor (legacy).
  val imgBufferSize = bandElements
  val imgDoubleBuffer = StreamDoubleBuffer(inputDataType, imgBufferSize, lanes = inLanes)
  // Re-arm boundary: rising edge of io.start.valid. Neither io.start.fire nor
  // dmaImg.io.cmd.fire is usable here — the synchronous fork only completes
  // its handshake when EVERY sink accepted, and the 2D image DMA accepts its
  // command on the LAST DRAINED BEAT, so both "fires" land after the image
  // has already filled a bank; re-arming then clears a fresh tileReady
  // forever (pipeline deadlock). The valid rising edge occurs one cycle
  // after the host pulses START, strictly before any DMA data moves.
  // With banding the same boundary re-arms the per-inference state (band
  // sequencer resets on it too via startTriggers, above).
  val prevStartValid = RegNext(io.start.valid) init (False)
  // S2c: the START edge re-arms for a new inference; a spill restartA pulse
  // re-arms for the next K-pass (same clear-and-refill semantics).
  // S2d scale lesson (K64-P8): do NOT re-arm the image buffer/streamer on
  // a spill restart. passDone fires after a few window beats while most of
  // the sweep is still in flight; re-arming then discards in-flight state
  // across components that do not all reset — the legacy 1->2 repack
  // adapter below ignores reArm and can hold one stale byte across the
  // boundary, permanently shifting every later pass's framing by one
  // element (silent wrong-GEMM at scale; short sweeps never trip it
  // because they fully drain before the cutoff). The re-fired sweeps carry
  // the SAME image, so the ping-pong self-synchronizes across passes
  // (full-tile delivery + tileReady gating, no boundary reset needed);
  // reArm stays for the inter-inference boundary (new image) only.
  // The band sequencer above still restarts per pass to re-issue the DMA
  // fill into the idle bank — that path is fully elastic.
  val imgReArm = (io.start.valid && !prevStartValid)
  imgDoubleBuffer.io.reArm := imgReArm
  imgDoubleBuffer.io.streamIn << dmaImg.io.outStream.stream

  val imgStreamer = DoubleBufferStreamer(inputDataType, imgBufferSize, lanes = inLanes)
  imgStreamer.io.readData := imgDoubleBuffer.io.readData
  imgStreamer.io.reArm := imgReArm
  imgStreamer.io.tileReady := imgDoubleBuffer.io.tileReady
  imgDoubleBuffer.io.readAddr := imgStreamer.io.readAddr
  imgDoubleBuffer.io.nextTile := imgStreamer.io.nextTile

  val imgQueue = Tensor(inputDataType, inputShape, inLanes)
  imgQueue.stream << imgStreamer.io.streamOut

  // --- 2. Node production ---
  // Every produced tensor becomes a graph node. Nodes consumed more than once
  // are forked; deferred branches flow through exact-capacity TapBuffers so a
  // one-shot inference never overflows them.
  val nodeOutputs = scala.collection.mutable.ArrayBuffer[scala.collection.mutable.ArrayBuffer[Tensor[Data]]]()

  def registerNode(raw: Tensor[Data]): Unit = {
    val cons = consumers(nodeOutputs.length)
    val views = if (cons.length > 1) TapBuffer.fork(raw, cons.length) else Seq(raw)
    nodeOutputs += scala.collection.mutable.ArrayBuffer(views: _*)
  }

  def inputFor(ref: Int, layerIdx: Int): Tensor[Data] =
    nodeOutputs(ref)(consumers(ref).indexOf(layerIdx))

  registerNode(imgQueue)

  for (i <- layers.indices) {
    val layer = layers(i)
    val lType = layer.outType(nodeTypes(i))
    val wType = layer.weightType(nodeTypes(i))

    val wShape = layer.getWeightShape()
    val bShape = layer.getBiasShape()
    // Fire of this layer's weight DMA (null when weightless): threaded into
    // matmul-based layers so their internal weight buffer re-arms per command.
    var weightDmaFire: Bool = null
    // Fire of the bias DMA (null when no bias): re-arms the bias cache
    // (BiasAddOp) at the command boundary so a stale generation bias cannot
    // contaminate the last tile of a pass.
    var biasDmaFire: Bool = null

    // S2b/S2c: the shared pass controller (created before the loop above)
    // is defined <=> a layer spills (v1 single layer). Its refetchW
    // re-fires reqW below; compute-side ports are wired at the compute site.

    var layerWeights: Tensor[Data] = null
    var layerBias: Tensor[Data] = null

    // Fetch Weights
    if (wShape.head > 0) {
      val elements = wShape.product
      // Per-beat weight width (M2 streaming): Linear/Conv1D/Conv2D expose
      // weightLanes (default = legacy full width); norm layers and attention
      // keep their structural widths (attention: wLanes == embedDim is a
      // hard require inside ClassicalAttentionHW — narrowing it belongs to a
      // dedicated change, not this knob).
      val requiredLanes = layer match {
        case c: Conv2D => c.effLanes
        case c: Conv1D => c.effLanes
        case l: Linear => l.effLanes
        case bn: BatchNorm1D => bn.features
        case ln: LayerNorm1D => ln.features
        case a: ClassicalAttention => a.embedDim
        case _ => 1
      }

      val dmaW = DMAReader(wType, wShape, outLanes = requiredLanes, dmaAxiConfig,
        trimToElements = true, flushableGearbox = true)
      // ---- Weight-residency + prefetch control plane (Phases 2a/2b) --------
      // STREAM_PER_PASS: byte-identical legacy behaviour incl. command-
      // boundary reArm. WEIGHT_RESIDENT: branches whose region is resident
      // swallow their START beat (no DDR); real fetches on first use, RELOAD
      // or the resident-mode rising edge (that edge self-fetches once: the
      // compute pointer usually sits on a flipped-empty bank). With
      // PREFETCH_EN added, such refresh fetches leave the START sweep
      // entirely: they fire EAGERLY against the reader-ready × loader-empty
      // intersection and stage a governed bank swap that lands at the NEXT
      // end-of-pass edge — never mid-stream. reArm is suppressed in the
      // prefetch world (held banks are live consumers); everything else
      // keeps the Phase-2a semantics verbatim.
      val fetchedOnceW = RegInit(False) init (False)
      val reloadPendingW = RegInit(False) init (False)
      weightReloadLatches += reloadPendingW
      val stagedW = RegInit(False) init (False)

      // Buffers/streamers FIRST: the eager arbitration observes loader capacity.
      // S2a spill slice: W[K][N] is row-major, so slice p (Ks rows x N) is
      // contiguous in DDR. A spilling layer fetches ONE slice per pass into
      // slice-sized buffers (the full-K mur is exactly what spill removes);
      // S2b re-fires this fetch per pass. Legacy layers keep full-region.
      // P1: any SpillableGEMM (Linear K=inFeatures, Conv2D K=K*K*inChannels).
      val spillSlice: Option[(Int, Int)] = layer match {
        case s: SpillableGEMM if s.spilling => Some((s.spillKSlice, s.spillPasses))
        case _ => None
      }
      val fetchElems = spillSlice match {
        case Some((ks, _)) => ks * wShape(1)
        case None => elements
      }
      // v1: every pass start must stay beat-aligned (the reader assumes
      // aligned starts — see the region-start contract above). Slices tile
      // the region with the EXACT slice stride (no per-slice alignment).
      val sliceBytes = spillSlice match {
        case Some((ks, _)) =>
          val raw = MemLayout.regionBytes(ks * wShape(1), wType.getBitsWidth)
          require(raw % beatBytes == 0,
            s"Sequential: spilling layer $i slice is $raw B, not a multiple of the AXI beat " +
              s"($beatBytes B) — every pass start must stay beat-aligned (v1 constraint: pick " +
              "spillKSlice with (spillKSlice*N*dtypeBits/8) a multiple of the beat)")
          raw
        case None => 0
      }
      val wBufferSize = fetchElems // exact-size contract (tile of `depth` elements)
      val wDoubleBuffer = StreamDoubleBuffer(wType, wBufferSize, requiredLanes,
        enableFreezePort = true)
      val wStreamer = DoubleBufferStreamer(wType, wBufferSize, requiredLanes)
      wStreamer.io.readData := wDoubleBuffer.io.readData
      wStreamer.io.tileReady := wDoubleBuffer.io.tileReady
      wDoubleBuffer.io.readAddr := wStreamer.io.readAddr
      wDoubleBuffer.io.nextTile := wStreamer.io.nextTile

      val prefetchWorldW = residentMode && io.weightPrefetch.getOrElse(False)
      val fetchNowW = !fetchedOnceW || !residentMode || reloadPendingW || residentRise
      val startPathW = startTriggers(triggerIdx).valid && fetchNowW
      val reqW = Stream(FetchRequest(axiConfig.addressWidth))
      // NN-01: `valid` must never depend on `ready`. The eager fetch is a pure
      // state function (sticky request x loader capacity); it holds until the
      // DMA accepts it and self-clears on `reqW.fire`.
      // S2b: W-slice refetch for spill passes > 0. The controller holds the
      // pulse until reqW fires (same sticky discipline); the slice address
      // follows spillPassIdx (S2a), advanced by the controller. No START
      // trigger is consumed (the fork sweep is long past).
      val spillRefetchW = spillCtrl match {
        case Some(c) => c.io.refetchW
        case None => False
      }
      reqW.valid := startPathW || spillRefetchW ||
        (prefetchWorldW && (reloadPendingW || residentRise) &&
          wDoubleBuffer.io.loadCanAccept && !startPathW)
      startTriggers(triggerIdx).ready := Mux(fetchNowW, reqW.ready, True)
      val wRegionOffset = alignToBeat(currentMemoryOffset)
      currentMemoryOffset = wRegionOffset
      // S2a: per-pass slice address. spillPassIdx stays 0 until the S2b
      // controller advances it; the formula already walks the slices so the
      // S2b diff only drives the register.
      val spillPassOff: UInt = spillSlice match {
        case Some((_, p)) =>
          // max 1: P == 1 (full-width single pass, legal) still needs a
          // well-formed register (same pattern as imgBandIdx above).
          // Driven by the S2b controller (single source of truth) at the
          // compute site below — no default assignment here (an unconditional
          // hold would overlap-error against the controller follow).
          val r = Reg(UInt((log2Up(p) max 1) bits)) init(0)
          spillPassIdxOf(i) = r
          spillPassIdxOf(i) = r
          (r * U(sliceBytes, axiConfig.addressWidth bits)).resize(axiConfig.addressWidth bits)
        case None => U(0, axiConfig.addressWidth bits)
      }
      // S2d-2 rerun lesson: a START-triggered fetch is ALWAYS pass 0, hence
      // slice 0 — never trust the pass-index register here. That register
      // follows ctrl.passIdx with a one-cycle delay, so at the START edge of
      // a second run it still holds the PREVIOUS run's final index (the
      // controller resets cnt on the same edge the follow samples the old
      // value). The run-2 pass-0 W fetch would then read the last slice
      // while the A window stays on slice 0 — silent wrong GEMM, y and the
      // spill region both shifted (RERUN replay). The mux is stable for
      // the whole sticky START request window (startPathW tracks
      // startTriggers.valid, which clears exactly on reqW.fire).
      val startSliceOffW = Mux(startPathW, U(0, axiConfig.addressWidth bits), spillPassOff)
      reqW.address := io.weightsBaseAddress + wRegionOffset + startSliceOffW
      val elementsPerBeatW = axiConfig.dataWidth / wType.getBitsWidth
      require(elementsPerBeatW >= 1,
        s"Weight dtype (${wType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported weight element size")
      val beats = (fetchElems + elementsPerBeatW - 1) / elementsPerBeatW
      spillSlice.foreach { case _ => spillSliceInfo(i) = (fetchElems, beats) }
      require(beats <= 65536,
        s"Weight ${spillSlice match { case Some(_) => "slice"; case None => "region" }} of layer $i needs $beats beats, above the 64K-beat single-command limit (multi-command fetch belongs to the multi-tile roadmap)")
      reqW.length := (if (beats > 0) beats - 1 else 0)
      dmaW.io.cmd << reqW

      allAxiMasters += dmaW.io.axiMaster
      triggerIdx += 1
      // Whole-region ceil via MemLayout: sub-byte weight types (e.g. I4
      // nibbles) must not floor to zero bytes, or the next region aliases
      // the weight region start.
      currentMemoryOffset = alignToBeat(currentMemoryOffset + MemLayout.regionBytes(elements, wType.getBitsWidth))

      weightDmaFire = reqW.fire
      when(reqW.fire) {
        fetchedOnceW := True
        reloadPendingW := False // RELOAD/rise/eager fetch consumed here
      }
      // Mode-off→on transition arms exactly one guaranteed refetch through
      // the SAME sticky latch RELOAD uses (a raw pulse here would be missed:
      // the rising edge lands between STARTs, long before the next trigger).
      when(residentRise) {
        reloadPendingW := True
      }
      // Prefetch world: suppress the destructive reArm (held banks are live
      // consumers); a staged swap is armed at fire and settles at the next
      // end-of-pass governed flip. Non-prefetch worlds keep Phase-2a exactly.
      when(reqW.fire && prefetchWorldW) {
        stagedW := True
      }
      wDoubleBuffer.io.reArm := reqW.fire && !prefetchWorldW
      wStreamer.io.reArm := reqW.fire
      wDoubleBuffer.io.residentHold.foreach(_ := residentMode)
      wDoubleBuffer.io.stageRequest.foreach(_ := stagedW)
      when(wDoubleBuffer.io.refreshSettled) {
        stagedW := False
      }
      wDoubleBuffer.io.streamIn << dmaW.io.outStream.stream

      layerWeights = Tensor(wType, wShape, requiredLanes)
      layerWeights.stream << wStreamer.io.streamOut
      // OPS-07: this weight stream is dense (exactly `elements` values, no
      // padding beats). It satisfies the MatMulOp per-line padded-group
      // contract iff the beat framing divides K: Linear/Conv1D/Conv2D enforce
      // lanes | K (`LayerSpec` require on weightLanes), so dense beats ==
      // column groups and no padding is needed. Any future producer with
      // K % lanes != 0 must zero-pad each line BEFORE this point, or the B
      // buffer starves.
      auditWeightLanes += requiredLanes
      auditWeightElements += elements
      auditWeightBeats += beats
    } else {
      auditWeightLanes += 0
      auditWeightElements += 0
      auditWeightBeats += 0
    }

    // Fetch Bias
    if (bShape.head > 0) {
      val requiredBiasLanes = layer match {
        case bn: BatchNorm1D => bn.features
        case ln: LayerNorm1D => ln.features
        case _ => 1
      }
      val elements = bShape.product
      val dmaB = DMAReader(lType, bShape, outLanes = requiredBiasLanes, dmaAxiConfig,
        trimToElements = true, flushableGearbox = true)
      // ---- Bias mirror of the weight prefetch/residency site (2a+2b) -------
      val fetchedOnceB = RegInit(False) init (False)
      val reloadPendingB = RegInit(False) init (False)
      weightReloadLatches += reloadPendingB
      val stagedB = RegInit(False) init (False)

      val bBufferSize = elements // exact size (contract = tile of `depth` elements)
      val bDoubleBuffer = StreamDoubleBuffer(lType, bBufferSize, requiredBiasLanes,
        enableFreezePort = true)
      val bStreamer = DoubleBufferStreamer(lType, bBufferSize, requiredBiasLanes)
      bStreamer.io.readData := bDoubleBuffer.io.readData
      bStreamer.io.tileReady := bDoubleBuffer.io.tileReady
      bDoubleBuffer.io.readAddr := bStreamer.io.readAddr
      bDoubleBuffer.io.nextTile := bStreamer.io.nextTile

      val prefetchWorldB = residentMode && io.weightPrefetch.getOrElse(False)
      val fetchNowB = !fetchedOnceB || !residentMode || reloadPendingB || residentRise
      val startPathB = startTriggers(triggerIdx).valid && fetchNowB
      val reqB = Stream(FetchRequest(axiConfig.addressWidth))
      // NN-01: `valid` must never depend on `ready` (bias mirror of the weight
      // eager-fetch site above).
      reqB.valid := startPathB ||
        (prefetchWorldB && (reloadPendingB || residentRise) &&
          bDoubleBuffer.io.loadCanAccept && !startPathB)
      startTriggers(triggerIdx).ready := Mux(fetchNowB, reqB.ready, True)
      currentMemoryOffset = alignToBeat(currentMemoryOffset)
      reqB.address := io.weightsBaseAddress + currentMemoryOffset

      val elementsPerBeatB = axiConfig.dataWidth / lType.getBitsWidth
      require(elementsPerBeatB >= 1,
        s"Bias dtype (${lType.getBitsWidth}b) is wider than the AXI beat (${axiConfig.dataWidth}b) — unsupported bias element size")
      val beatsB = (elements + elementsPerBeatB - 1) / elementsPerBeatB
      require(beatsB <= 65536,
        s"Bias region of layer $i needs $beatsB beats, above the 64K-beat single-command limit (multi-command fetch belongs to the multi-tile roadmap)")
      reqB.length := (if (beatsB > 0) beatsB - 1 else 0)

      dmaB.io.cmd << reqB

      allAxiMasters += dmaB.io.axiMaster
      triggerIdx += 1
      currentMemoryOffset = alignToBeat(currentMemoryOffset + MemLayout.regionBytes(elements, lType.getBitsWidth))

      when(reqB.fire) {
        fetchedOnceB := True
        reloadPendingB := False
      }
      when(residentRise) {
        reloadPendingB := True
      }
      when(reqB.fire && prefetchWorldB) {
        stagedB := True
      }
      biasDmaFire = reqB.fire
      bDoubleBuffer.io.reArm := reqB.fire && !prefetchWorldB
      bStreamer.io.reArm := reqB.fire
      bDoubleBuffer.io.residentHold.foreach(_ := residentMode)
      bDoubleBuffer.io.stageRequest.foreach(_ := stagedB)
      when(bDoubleBuffer.io.refreshSettled) {
        stagedB := False
      }
      bDoubleBuffer.io.streamIn << dmaB.io.outStream.stream

      layerBias = Tensor(lType, bShape, requiredBiasLanes)
      layerBias.stream << bStreamer.io.streamOut
    }

    // Instantiate computation block
    val inTensor = inputFor(i, i)

    val nextTensor: Tensor[Data] = layer match {
      case c: Conv1D =>
        Conv1DHW(inTensor, layerWeights, layerBias, lType, reArm = Option(weightDmaFire), temporal = temporal, outLanes = c.lanes)

      case c: Conv2D =>
        // Integer-domain convolutions: narrow SInt weights (e.g. true I4
        // nibbles fetched from DDR) are sign-extended to the activation width
        // so the single-width int matmul can consume them.
        val wDT = layerWeights.dataType()
        val aDT = inTensor.dataType()
        val wForConv =
          if (wDT.isInstanceOf[SInt] && aDT.isInstanceOf[SInt] && wDT.getBitsWidth != aDT.getBitsWidth) {
            require(wDT.getBitsWidth < aDT.getBitsWidth,
              s"Conv2D weights (${wDT.getBitsWidth}b) wider than activations (${aDT.getBitsWidth}b): " +
                "narrowing auto-cast refused — insert an explicit Cast layer or use wider activations")
            cast(layerWeights.asInstanceOf[Tensor[SInt]], inTensor.dataType)
          } else {
            require(!(aDT.isInstanceOf[FloatML] && wDT.isInstanceOf[SInt]),
              "Conv2D with float activations and integer weights is unsupported (float dequant is Linear-only): " +
                "quantize the activations or run this stage on integer activations")
            layerWeights
          }
        // P1 spill sizing (mirror of the Linear block below): the K-pass GEMM
        // needs the windowed row drain (temporal) and a re-streamable A every
        // pass (node 0 DDR-backed re-fire, or on-chip StreamTap replay within
        // budget); M*N full-width partials sized here (M = total windows).
        if (c.spilling) {
          require(temporal >= 1,
            s"Sequential: Conv2D layer $i spills (spillKSlice=${c.spillKSlice}) but temporal=$temporal — " +
              "the spill drain reuses the windowed row drain, require temporal >= 1")
          val aElems = nodeShapes(i).product
          val aBytes = MemLayout.regionBytes(aElems, nodeTypes(i).getBitsWidth)
          require((i == 0 && consumers(0).size == 1) || aBytes <= spillReplayBudgetBytes,
            s"Sequential: Conv2D layer $i spills but its A operand (node $i, ${aBytes}B) is neither " +
              "exclusively DDR-backed (sole consumer of node 0, re-fired per pass) nor within " +
              "spillReplayBudgetBytes=$spillReplayBudgetBytes (StreamTap replay) — " +
              "spill A to DDR first (v2, see docs/ddr_final_impl.md)")
          val mRows = (nodeShapes(i)(0) - c.kernelSize + 1) * (nodeShapes(i)(1) - c.kernelSize + 1)
          val spillElems = mRows * c.outChannels
          spillBytesAcc += alignToBeat(MemLayout.regionBytes(spillElems, lType.getBitsWidth))
        }
        // P1 K-pass wiring (mirror of the Linear S2b block above): direct
        // Conv2DLayer instantiation (not via apply) so the pass controller
        // owns the spill ports. SLICE GEOMETRY (S1 proven): the engine is
        // shaped [M,Ks]x[Ks,N] per pass with M = total windows; the A-window
        // lives inside Conv2DLayer on the im2col cols stream (same
        // accept-and-drop discipline, passIdx held by the controller).
        val convOut: Tensor[Data] = if (c.spilling) {
          // A re-stream: exclusive node 0 re-fires the image sweep from DDR
          // (im2col reproduces cols verbatim per pass, self-restarting);
          // shared/deep nodes replay through an on-chip StreamTap.
          val inFed: Tensor[Data] = if (!spillImgRefire) {
            val tap = StreamTap(nodeTypes(i), nodeShapes(i), nodeLanes(i))
            tap.io.arm := io.start.valid && !prevStartValid
            tap.io.replay := spillCtrl.get.io.restartA
            tap.io.streamIn.stream << inTensor.stream
            val fed = Tensor(nodeTypes(i), nodeShapes(i), nodeLanes(i))
            fed.stream << tap.io.streamOut.stream
            fed
          } else inTensor
          val hIn = nodeShapes(i)(0)
          val wIn = nodeShapes(i)(1)
          val chIn = if (nodeShapes(i).length >= 3) nodeShapes(i)(2) else 1
          val mRows = (hIn - c.kernelSize + 1) * (wIn - c.kernelSize + 1)
          val layerSpillOffset = spillBytesAcc
          val convComp = spinalML.layers.Conv2DLayer(
            inFed.dataType, lType, hIn, wIn, chIn, c.outChannels, c.kernelSize,
            outLanes = c.effLanes, temporal = temporal, inLanes = inFed.lanes,
            convOutLanes = c.lanes, spill = true, spillKSlice = c.spillKSlice)
          convComp.io.reArm := weightDmaFire
          convComp.io.x.stream << inFed.stream
          convComp.io.w.stream << wForConv.stream
          convComp.io.b <> layerBias
          val ctrl = spillCtrl.get // Some <=> spilling (v1 single layer)
          val spillWriter = DMAWriter(lType, Seq(mRows, c.outChannels), 1, spillWriteLeafCfg)
          val spillReader = DMAReader(lType, Seq(mRows, c.outChannels), 1, dmaAxiConfig,
            trimToElements = true, flushableGearbox = true)
          allAxiMasters += spillReader.io.axiMaster
          io.spillWrite.get <> spillWriter.io.axiMaster
          // Pass-loop wiring: identical contract to the Linear block.
          convComp.io.passFirst.get := ctrl.io.passFirst
          convComp.io.passLast.get := ctrl.io.passLast
          convComp.io.passIdx.get := spillPassIdxOf(i)
          convComp.io.spillIn.get.stream << spillReader.io.outStream.stream
          spillWriter.io.inStream.stream << convComp.io.spillOut.get.stream
          spillReader.io.cmd << ctrl.io.readerCmd
          spillWriter.io.cmd << ctrl.io.writerCmd
          ctrl.io.passDone := convComp.io.passDone.get
          ctrl.io.writerDone := spillWriter.io.done
          ctrl.io.wFetchFire := weightDmaFire
          ctrl.io.residentMode := residentMode
          ctrl.io.spillBase := io.spillBaseAddress + layerSpillOffset
          ctrl.io.start := io.start.valid && !prevStartValid
          // Per-pass bias re-arm WITHOUT bias re-fetch (final prelude only).
          convComp.io.biasReArm := biasDmaFire || (ctrl.io.biasReArm && ctrl.io.passLast)
          // The S2a pass index follows the controller (single source of
          // truth for the slice addressing in the fetch plane above).
          spillPassIdxOf(i) := ctrl.io.passIdx
          convComp.io.y
        } else {
          Conv2DHW(inTensor, wForConv, layerBias, lType, reArm = Option(weightDmaFire), temporal = temporal, outLanes = c.lanes)
        }
        convOut

      case _: ReLU =>
        relu(inTensor)

      case lr: LeakyReLU =>
        leaky_relu(inTensor, lr.shift)

      case sm: Softmax =>
        val seqLen = nodeShapes(i)(0)
        val channels = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val comp = spinalML.activations.Softmax1D(nodeTypes(i), channels, seqLen)
        comp.io.x <> (if (inTensor.lanes != channels) repack(inTensor, channels) else inTensor)
        val smOut = comp.io.y
        if (smOut.lanes != sm.lanes) repack(smOut, sm.lanes) else smOut

      case bn: BatchNorm1D =>
        val targetLanes = if (bn.lanes > 0) bn.lanes else bn.features
        val inRepacked = if (inTensor.lanes != bn.features) repack(inTensor, bn.features) else inTensor
        val bnOut = batchnorm(inRepacked, layerWeights, layerBias,
          reArm = Option(weightDmaFire), shift = bn.shift,
          rounding = bn.rounding.getOrElse(spinalML.RoundingConfig.current))
        if (bnOut.lanes != targetLanes) repack(bnOut, targetLanes) else bnOut

      case ln: LayerNorm1D =>
        val seqLen = nodeShapes(i)(0)
        val targetLanes = if (ln.lanes > 0) ln.lanes else ln.features
        val inRepacked = if (inTensor.lanes != ln.features) repack(inTensor, ln.features) else inTensor
        val comp = spinalML.layers.LayerNorm1D(nodeTypes(i), ln.features, seqLen)
        comp.io.x <> inRepacked
        comp.io.gamma <> layerWeights
        comp.io.beta <> layerBias
        comp.io.reArm := weightDmaFire
        val lnOut = comp.io.y
        if (lnOut.lanes != targetLanes) repack(lnOut, targetLanes) else lnOut

      case mp: MaxPool1D =>
        val c = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val targetLanes = if (mp.lanes > 0) mp.lanes else c
        val repacked = if (inTensor.lanes != c) repack(inTensor, c) else inTensor
        val pooled = maxpool1d(repacked, mp.poolSize, mp.stride)
        if (pooled.lanes != targetLanes) repack(pooled, targetLanes) else pooled

      case ap: AvgPool1D =>
        val c = if (nodeShapes(i).length > 1) nodeShapes(i)(1) else 1
        val targetLanes = if (ap.lanes > 0) ap.lanes else c
        val repacked = if (inTensor.lanes != c) repack(inTensor, c) else inTensor
        val pooled = avgpool1d(repacked, ap.poolSize, ap.stride,
          rounding = ap.rounding.getOrElse(spinalML.RoundingConfig.current))
        if (pooled.lanes != targetLanes) repack(pooled, targetLanes) else pooled

      case mp2: MaxPool2D =>
        val pooled = maxpool2d(inTensor, mp2.poolSize, mp2.stride)
        if (pooled.lanes != mp2.lanes) repack(pooled, mp2.lanes) else pooled

      case ap2: AvgPool2D =>
        val pooled = avgpool2d(inTensor, ap2.poolSize, ap2.stride,
          rounding = ap2.rounding.getOrElse(spinalML.RoundingConfig.current))
        if (pooled.lanes != ap2.lanes) repack(pooled, ap2.lanes) else pooled

      case sm: Sigmoid =>
        sigmoid(inTensor, sm.inputScale, sm.inputZeroPoint,
          rounding = sm.rounding.getOrElse(spinalML.RoundingConfig.current))

      case th: Tanh =>
        tanh(inTensor, th.inputScale, th.inputZeroPoint,
          rounding = th.rounding.getOrElse(spinalML.RoundingConfig.current))

      case c: Cast =>
        val scalePort = if (c.runtimeScale) io.dequantScale else None
        cast(inTensor, lType, c.scales, runtimeScalePort = scalePort,
          rounding = c.rounding.getOrElse(spinalML.RoundingConfig.current))

      case _: Flatten =>
        reshape(flatten(inTensor), Seq(1, inTensor.shape.product))

      case l: Linear =>
        // S2c A re-stream: an exclusive node-0 spill re-fires DDR in the
        // image plane above; a shared node 0 or a deep node replays this
        // branch through an on-chip tap (snoop pass 0, verbatim replay on
        // each restartA pulse — the replay flows through the SAME reshape +
        // repack below, bit-exact by construction).
        val inFed: Tensor[Data] = if (l.spilling && !spillImgRefire) {
          val tap = StreamTap(nodeTypes(i), nodeShapes(i), nodeLanes(i))
          tap.io.arm := io.start.valid && !prevStartValid
          tap.io.replay := spillCtrl.get.io.restartA
          tap.io.streamIn.stream << inTensor.stream
          val fed = Tensor(nodeTypes(i), nodeShapes(i), nodeLanes(i))
          fed.stream << tap.io.streamOut.stream
          fed
        } else inTensor
        val rows = inFed.shape.dropRight(1).product
        val reshaped = reshape(inFed, Seq(rows, l.inFeatures))
        // DO NOT switch this repack to withFlush = true without a local
        // elastic stage (FIFO >= 2 or a pipe pair) at this fan-out attach
        // point: the flushable gearbox's hard `ready := !full` chained
        // combinationally onto the node0 tee corrupted the OTHER fork branch
        // (skip-FIFO lost/duplicated the boundary element — ResidualMLP).
        // Bisection evidence + elasticity rule: docs/open-mysteries.md M1.7.
        // M2: the beat width is `weightLanes` (<= inFeatures); the matmul
        // accumulates the K chunks internally in order, so bit-exactness is
        // preserved as long as the oracle reproduces the same chunk fold
        // (MnistReplica.linearLayer wLanes).
        val repackedTensor = repack(reshaped, l.effLanes)
        // Weight-only quantization (wXaY): SInt weights (I4/I8) + compile-time scale(s)
        // are dequantized to the activation float dtype inside the layer.
        val linOut: Tensor[Data] = if (l.spilling) {
          // S2b K-pass wiring (docs/ddr_final_impl.md): direct LinearLayer
          // instantiation (not via apply) so the pass controller owns the
          // spill ports. One uniform call for both dtype paths — same-dtype
          // and wXaY dequant decide inside the component (weightScales
          // default Seq(1.0) keeps same-dtype exact).
          // SLICE GEOMETRY (S1 proven): the engine is shaped [M,Ks]x[Ks,N]
          // per pass — full shapes would starve its shape-driven B/A
          // counters on partial streams. W already streams slices (S2a
          // fetch); A re-streams WHOLE (re-fire/tap) and is windowed here.
          // S2c K-window on A: pass p consumes cols [p*Ks,(p+1)*Ks) = beats
          // [p*B1,(p+1)*B1) of each R-beat row (beat-aligned by S0
          // Ks%effLanes==0). Non-window beats are accepted-and-dropped so
          // upstream never stalls; passIdx is prelude-stable during flow.
          val rowBeatsA = l.inFeatures / l.effLanes
          val winBeatsA = l.spillKSlice / l.effLanes
          val aWinT = Tensor(repackedTensor.dataType, repackedTensor.shape, repackedTensor.lanes)
          val aBeatInRow = Reg(UInt((log2Up(rowBeatsA) max 1) bits)) init(0)
          val winLo = (spillPassIdxOf(i) * U(winBeatsA, 16 bits)).resize(16 bits)
          val inWin = aBeatInRow.resize(16 bits) >= winLo &&
            aBeatInRow.resize(16 bits) < winLo + U(winBeatsA, 16 bits)
          // S2d debug probes (see member maps above).
          winLo.simPublic()
          aBeatInRow.simPublic()
          spillWinLoOf(i) = winLo
          spillABeatOf(i) = aBeatInRow
          aWinT.stream.valid := repackedTensor.stream.valid && inWin
          aWinT.stream.payload := repackedTensor.stream.payload
          repackedTensor.stream.ready := !inWin || aWinT.stream.ready
          when(repackedTensor.stream.fire) {
            when(aBeatInRow === U(rowBeatsA - 1, (log2Up(rowBeatsA) max 1) bits)) {
              aBeatInRow := 0
            } otherwise {
              aBeatInRow := aBeatInRow + 1
            }
          }
          // Slice views (stream aliases, S1 geometry): windowed A [M,Ks],
          // fetched W slice [Ks,N]. Beat counts match the flows exactly.
          val aSlice = Tensor(repackedTensor.dataType, Seq(rows, l.spillKSlice), repackedTensor.lanes)
          aSlice.stream << aWinT.stream
          val wSlice = Tensor(layerWeights.dataType, Seq(l.spillKSlice, l.outFeatures), repackedTensor.lanes)
          wSlice.stream << layerWeights.stream
          // The spill region offset of this layer (single region per layer,
          // captured before the += in the sizing block below).
          val layerSpillOffset = spillBytesAcc
          val linComp = spinalML.layers.LinearLayer(repackedTensor.dataType, layerWeights.dataType, lType,
            Seq(rows, l.spillKSlice), Seq(l.spillKSlice, l.outFeatures), repackedTensor.lanes,
            l.weightScales, 1024, false, temporal, spill = true)
          linComp.io.reArm := weightDmaFire
          linComp.io.a.stream << aSlice.stream
          linComp.io.w.stream << wSlice.stream
          linComp.io.b <> layerBias
          val ctrl = spillCtrl.get // Some <=> spilling (v1 single layer)
          // M*N full-width partial geometries (accType = lType, as noted in
          // the sizing block below; beats match the controller's command).
          val mRows = nodeShapes(i).dropRight(1).product
          val spillWriter = DMAWriter(lType, Seq(mRows, l.outFeatures), 1, spillWriteLeafCfg)
          val spillReader = DMAReader(lType, Seq(mRows, l.outFeatures), 1, dmaAxiConfig,
            trimToElements = true, flushableGearbox = true)
          allAxiMasters += spillReader.io.axiMaster
          io.spillWrite.get <> spillWriter.io.axiMaster
          // Pass-loop wiring: the controller owns both spill cmd streams;
          // partials flow compute -> DDR (drain) and DDR -> compute (seed).
          linComp.io.passFirst.get := ctrl.io.passFirst
          linComp.io.passLast.get := ctrl.io.passLast
          linComp.io.spillIn.get.stream << spillReader.io.outStream.stream
          spillWriter.io.inStream.stream << linComp.io.spillOut.get.stream
          spillReader.io.cmd << ctrl.io.readerCmd
          spillWriter.io.cmd << ctrl.io.writerCmd
          ctrl.io.passDone := linComp.io.passDone.get
          ctrl.io.writerDone := spillWriter.io.done
          // weightDmaFire IS this layer's reqW.fire (hoisted var, same
          // signal): the pass-0 START fetch and every controller refetch
          // both pulse it — the S1 pass-fire edge rides for free.
          ctrl.io.wFetchFire := weightDmaFire
          ctrl.io.residentMode := residentMode
          ctrl.io.spillBase := io.spillBaseAddress + layerSpillOffset
          ctrl.io.start := io.start.valid && !prevStartValid
          // Per-pass bias re-arm WITHOUT bias re-fetch: the bias stays
          // parked from the START fetch and is consumed on the final pass
          // only (non-final passes feed the on-chip zero mux, S1).
          // S2c e2e lesson: gate the controller pulse to the FINAL prelude.
          // An every-prelude pulse can catch BiasAddOp mid-load across a
          // passLast flip, letting a stale partial load complete with mixed
          // zero/real beats. The final-only pulse always finds settled
          // levels (loaded at entry) and a parked bias, so its reload is
          // atomic-real; earlier episodes (reset-auto, bias-fetch) always
          // complete with stable levels too.
          linComp.io.biasReArm := biasDmaFire || (ctrl.io.biasReArm && ctrl.io.passLast)
          // The S2a pass index follows the controller (single source of
          // truth for the slice addressing in the fetch plane above).
          spillPassIdxOf(i) := ctrl.io.passIdx
          linComp.io.y
        } else layerWeights.dataType() match {
          case _: SInt =>
            spinalML.layers.Linear(repackedTensor, layerWeights.asInstanceOf[Tensor[SInt]], layerBias, lType, l.weightScales,
              false, 1024, Option(weightDmaFire), Option(biasDmaFire), temporal)
          case _ =>
            LinearHW(repackedTensor, layerWeights, layerBias, lType, 1024, false, Option(weightDmaFire), Option(biasDmaFire), temporal)
        }
        // The K-pass GEMM needs the windowed row drain (temporal) and must
        // be able to re-stream A every pass: node 0 is DDR-backed
        // (re-fetch), deeper nodes need a replay buffer within budget
        // (A-spill to DDR is the v2 follow-up, see docs/ddr_final_impl.md).
        // NOTE (runtime contract, enforced by the S2 pass controller): spill
        // passes assume STREAM_PER_PASS (CSR 0x10 = 0). The residency control
        // plane may be wired (weightResidency flag) as long as the host never
        // enables resident/prefetch modes under a spill — hence no
        // elaboration require on the flag itself (Accelerator enables it by
        // default).
        if (l.spilling) {
          require(temporal >= 1,
            s"Sequential: Linear layer $i spills (spillKSlice=${l.spillKSlice}) but temporal=$temporal — " +
              "the spill drain reuses the windowed row drain, require temporal >= 1")
          val aElems = nodeShapes(i).product
          val aBytes = MemLayout.regionBytes(aElems, nodeTypes(i).getBitsWidth)
          // S2c: exclusive node 0 re-fires DDR (no budget needed); shared
          // node 0 and deep nodes replay from an on-chip StreamTap within
          // budget (A-spill to DDR is the v2 follow-up).
          require((i == 0 && consumers(0).size == 1) || aBytes <= spillReplayBudgetBytes,
            s"Sequential: Linear layer $i spills but its A operand (node $i, ${aBytes}B) is neither " +
              "exclusively DDR-backed (sole consumer of node 0, re-fired per pass) nor within " +
              "spillReplayBudgetBytes=$spillReplayBudgetBytes (StreamTap replay) — " +
              "spill A to DDR first (v2, see docs/ddr_final_impl.md)")
          // M*N full-width partials (accType = lType at both call sites above).
          val mRows = nodeShapes(i).dropRight(1).product
          val spillElems = mRows * l.outFeatures
          spillBytesAcc += alignToBeat(MemLayout.regionBytes(spillElems, lType.getBitsWidth))
        }
        if (linOut.lanes != l.lanes) repack(linOut, l.lanes) else linOut

      case rq: Requantize =>
        spinalML.ops.requantize(inTensor, rq.targetType, rq.shift, rq.rounding.getOrElse(spinalML.RoundingConfig.current))

      case rp: Repack =>
        repack(inTensor, rp.newLanes)

      case ad: Add =>
        val ta = inputFor(ad.a, i)
        val tbRaw = inputFor(ad.b, i)
        val tb = if (tbRaw.lanes != ta.lanes) repack(tbRaw, ta.lanes) else tbRaw
        spinalML.ops.add(ta, tb)

      case cc: Concat =>
        val ta0 = inputFor(cc.a, i)
        val tb0 = inputFor(cc.b, i)
        // OPS-04: ConcatenateAxis0Op counts streamed beats (one axis-0 cell
        // = tailProduct/lanes beats). Repack both inputs to one full row per
        // beat so the beat count is exactly L_A + L_B (bit-exact against the
        // universal replica concat).
        val rowLanes = ta0.shape.drop(1).product
        val ta = if (ta0.lanes != rowLanes) repack(ta0, rowLanes) else ta0
        val tb = if (tb0.lanes != rowLanes) repack(tb0, rowLanes) else tb0
        val catOut = spinalML.ops.concatenate(ta, tb, 0)
        if (catOut.lanes != cc.lanes) repack(catOut, cc.lanes) else catOut

      case a: ClassicalAttention =>
        val seqLen = nodeShapes(i)(0)
        // wType carries the (possibly quantized) weight dtype declared via
        // customWeightType; scales drive the in-layer dequantization.
        val comp = ClassicalAttentionHW(nodeTypes(i), wType, lType, seqLen, a.embedDim, a.numHeads, inTensor.lanes, layerWeights.lanes, weightScales = a.weightScales)
        comp.io.x <> inTensor

        // Fork and slice the weights stream into 4 parts
        val wForks = StreamFork(layerWeights.stream, 4)

        val w0 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w1 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w2 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        val w3 = Tensor(layerWeights.dataType, layerWeights.shape, layerWeights.lanes)
        w0.stream << wForks(0)
        w1.stream << wForks(1)
        w2.stream << wForks(2)
        w3.stream << wForks(3)

        comp.io.wq <> spinalML.ops.slice(w0, 0, a.embedDim, axis = 0)
        comp.io.wk <> spinalML.ops.slice(w1, a.embedDim, 2 * a.embedDim, axis = 0)
        comp.io.wv <> spinalML.ops.slice(w2, 2 * a.embedDim, 3 * a.embedDim, axis = 0)
        comp.io.wo <> spinalML.ops.slice(w3, 3 * a.embedDim, 4 * a.embedDim, axis = 0)
        val rawY = comp.io.y
        if (rawY.lanes != a.lanes) repack(rawY, a.lanes) else rawY
    }

    registerNode(nextTensor)
  }

  // Total weight/bias region footprint in bytes (exact `MemLayout`
  // conventions: whole-region ceil + beat alignment per region). Phase-2 DDR
  // plumbing: `Accelerator` uses this for the elaboration-time fit check
  // (`MemorySpec.reportFit`) instead of duplicating the layout loop.
  val totalWeightBytes: Int = currentMemoryOffset

  // S0 compute-side spill footprint in bytes (exact `MemLayout` conventions:
  // per-region ceil + beat alignment). 0 = no spilling layer. `Accelerator`
  // feeds this to `MemorySpec.reportFit` instead of the declared hint.
  val totalSpillBytes: Int = spillBytesAcc

  // RELOAD broadcast (Phase 2a weight residency): a pulse on this input arms
  // EVERY resident region for exactly one refetch at the next START. Placed
  // textually after the per-command latch-clear sites so it wins there
  // (last-assignment-wins semantics).
  when(io.weightReload.getOrElse(False)) {
    weightReloadLatches.foreach(_ := True)
  }

  // Output assignment: the last node feeds the accelerator output stream
  io.outStream <> nodeOutputs.last(consumers.last.indexOf(-1))

  // ---- Frame accounting (Phase 3, S1) -----------------------------------
  // A free-running modulo-frameSize counter of the final stream fires is
  // exact because consecutive frames are contiguous by construction (a
  // complete inference = exactly frameSize beats; a frame never stalls
  // partially at handshake level — fires only count completed beats).
  val frameSize = (finalShape.product + finalLanes - 1) / finalLanes
  require(frameSize > 0, "Sequential: output frame must be non-empty")
  val frameCounter = Counter(frameSize)
  when(io.outStream.stream.fire) {
    frameCounter.increment()
  }
  val ioBusy = RegInit(False)
  // NN-02: a START accepted on the same cycle as the final-beat clear must win
  // (last-assignment-wins would drop busy while a new inference just began).
  when(io.start.fire) {
    ioBusy := True
  } elsewhen(ioBusy && io.outStream.stream.fire && frameCounter.willOverflowIfInc) {
    ioBusy := False
  }
  io.busy := ioBusy
  io.done := ioBusy && io.outStream.stream.fire && frameCounter.willOverflowIfInc

  // --- 3. AXI Read Arbitration ---
  // If only a single DMA exists (e.g. image-only weightless model), bypass the arbiter entirely.
  // Up to arbFanIn masters share one single-stage arbiter; beyond that a
  // two-stage tree (groups of <= arbFanIn, then a root arbiter) bounds the
  // per-stage fan-in for timing and removes the single-stage ID squeeze.
  if (allAxiMasters.length == 1) {
    io.axiMaster <> allAxiMasters.head
  } else if (allAxiMasters.length <= arbFanIn) {
    val arbiter = Axi4ReadOnlyArbiter(axiConfig, allAxiMasters.length)
    for (i <- allAxiMasters.indices) {
      arbiter.io.inputs(i) <> allAxiMasters(i)
    }
    io.axiMaster <> arbiter.io.output
  } else {
    // Every stage-1 arbiter is sized to arbFanIn (unused inputs tied
    // request-silent) so all group outputs share one ID width into root:
    // dmaAxiConfig.idWidth + log2Up(arbFanIn) == axiConfig.idWidth - log2Up(groups).
    val stage1OutCfg = axiConfig.copy(idWidth = dmaAxiConfig.idWidth + log2Up(arbFanIn))
    val groupOutputs = scala.collection.mutable.ArrayBuffer[Axi4ReadOnly]()
    var idx = 0
    while (idx < allAxiMasters.length) {
      val arb = Axi4ReadOnlyArbiter(stage1OutCfg, arbFanIn)
      for (k <- 0 until arbFanIn) {
        if (idx + k < allAxiMasters.length) {
          arb.io.inputs(k) <> allAxiMasters(idx + k)
        } else {
          arb.io.inputs(k).ar.valid := False
          arb.io.inputs(k).ar.payload.assignDontCare()
          arb.io.inputs(k).r.ready := True
        }
      }
      groupOutputs += arb.io.output
      idx += arbFanIn
    }
    val root = Axi4ReadOnlyArbiter(axiConfig, groupOutputs.length)
    for (i <- groupOutputs.indices) {
      root.io.inputs(i) <> groupOutputs(i)
    }
    io.axiMaster <> root.io.output
  }

  // ---- SimLog audit: model summary (INFO) + per-node table (DEBUG) --------
  SimLog.info("MODEL")(s"Sequential: $nNodes nodes, " +
    s"dtypes=${(0 until nNodes).map(i => hardTypeName(nodeTypes(i))).mkString(" -> ")}")
  SimLog.info("MODEL")(s"shapes: ${nodeShapes.map(_.mkString("x")).mkString(" -> ")}; " +
    s"outElems=${finalShape.product}, weightElems=${auditWeightElements.sum}, " +
    s"weightLanesMax=${if (auditWeightLanes.nonEmpty) auditWeightLanes.max else 0}")
  if (auditWeightLanes.nonEmpty && auditWeightLanes.max > 64)
    SimLog.warn("MODEL")(s"weightLanes=${auditWeightLanes.max} exceeds the 64-lane sanity threshold — large LUT footprint " +
      s"(lane decomposition is the M2 roadmap item)")
  if (SimLog.isDebug) {
    SimLog.debug("MODEL")(s"n0 INPUT shape=${inputShape.mkString("x")} dtype=${hardTypeName(nodeTypes(0))}")
    for (i <- layers.indices)
      SimLog.debug("MODEL")(s"n${i + 1} ${layers(i).getClass.getSimpleName} in=${nodeShapes(i).mkString("x")} " +
        s"out=${nodeShapes(i + 1).mkString("x")} dtype=${hardTypeName(nodeTypes(i + 1))} " +
        s"elems=${nodeShapes(i + 1).product} wLanes=${auditWeightLanes(i)} " +
        s"wElems=${auditWeightElements(i)} wBeats=${auditWeightBeats(i)}")
  }
}
