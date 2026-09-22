// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinalML.tensors.Tensor
import spinalML.memory.StreamDoubleBuffer

/**
 * MatmulOp: Matrix-Matrix multiplication using Double-Buffering (Ping-Pong).
 * A is [M, K], B is [K, N].
 * Output C is [M, N].
 *
 * Streaming contract (OPS-07 / OPS-09): both inputs arrive in per-line padded
 * groups of exactly chunksK = ceil(K / lanes) full beats — per ROW for A,
 * per COLUMN (column-major) for B. The tail lanes of a partial group are
 * don't-care: the compute datapath masks them to zero (`validLane`), so
 * zero-padding and garbage-padding behave identically. A densely packed
 * stream (exactly K elements per line, no padding beats) is OUT of contract
 * when K % lanes != 0: its beats straddle line boundaries, the B buffer
 * starves and the op deadlocks. Producers pad (cf. `test_matmul.py` driver,
 * Sequential Dense paths with lanes | K); partial-K handling inside the op
 * is deliberately absent — padding at the producer is the contract.
 */
case class MatmulOp[T <: Data, TAcc <: Data](
  dataType: HardType[T],
  accType: HardType[TAcc],
  shapeA: Seq[Int],
  shapeB: Seq[Int],
  lanes: Int,
  parallelN: Boolean = false,
  pipelineTree: Boolean = true,
  // Rows-in-flight bound for the output accumulator table. 0 = legacy: the
  // whole MxN partial table is materialized (large registers + index mux).
  // > 0: the accumulation is drained row-by-row as soon as each row is
  // complete, so the table shrinks to min(temporal, M) x N slots — the sum
  // order (and therefore bit-exactness) is unchanged; only storage shrinks.
  temporal: Int = 0,
  // S1 compute-side spill (docs/ddr_final_impl.md): K-pass slice engine. The
  // op is deliberately slice-agnostic — each pass looks like one self-
  // contained GEMM over the slice geometry (A [M, Ks], B [Ks, N]) chained by
  // the spill streams: pass 0 seeds zeros, passes > 0 seed from spillIn,
  // non-final passes drain M*N partials to spillOut, the final pass drains
  // to io.c. The pass LOOP (re-fire, slice addresses, pass counting) lives
  // in Sequential (S2); this op only exposes passFirst/passLast levels
  // (held stable by the controller from pass-fire to passDone) and pulses
  // passDone on every pass-drain completion.
  // spillPadElems: trailing pad elements closing the seed region's last AXI
  // beat (region beats cover ceil(M*N*accBytes/beatBytes) beats; the pad is
  // M*N's complement to whole beats). The S2 controller fetches the seed in
  // single-beat chunks paced by the flushable reader's accept gate, so every
  // fetched beat — pad included — lands in the reader gearbox; on non-first
  // passes the engine drop-drains exactly spillPadElems beats after the last
  // row's emit, before pulsing passDone. Without the drain the pad would sit
  // in the gearbox, blocking the next pass's first chunk (accept gate) or
  // poisoning its row-0 seed. 0 = beat-exact region (no drain phase).
  spill: Boolean = false,
  spillPadElems: Int = 0,
  dspConfig: spinalML.dsp.DspConfig = spinalML.dsp.DspConfig.default
) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  val N = shapeB(1)
  require(shapeB(0) == K, "Inner dimensions must match (A.cols == B.rows)")
  require(temporal >= 0, s"temporal=$temporal must be >= 0")
  require(temporal == 0 || !parallelN,
    s"temporal=$temporal requires the sequential-N matmul (parallelN=false)")
  require(!spill || !parallelN,
    s"spill=true requires the sequential-N matmul (parallelN=false)")
  require(!spill || temporal >= 1,
    s"spill=true requires temporal >= 1 (the spill drain reuses the windowed row drain)")
  require(spillPadElems >= 0,
    s"spillPadElems=$spillPadElems must be >= 0")
  require(spill || spillPadElems == 0,
    s"spillPadElems=$spillPadElems without spill=true (nothing fetches a pad)")
  
  val chunksK = (K + lanes - 1) / lanes
  val paddedK = chunksK * lanes
  val treeLatency = if (pipelineTree) log2Up(lanes) else 0

  val io = new Bundle {
    val a = slave(Tensor(dataType, shapeA, lanes))
    val b = slave(Tensor(dataType, shapeB, lanes))
    val c = master(Tensor(accType, Seq(M, N), lanes = 1))
    // Command-boundary re-arm forwarded to the internal B buffer(s): without
    // it, a stale tileReady lets the next command start on the previous
    // command's data. See StreamDoubleBuffer.io.reArm.
    val reArm = in Bool()
    // S1 spill ports (spill=true only): M*N full-width partials, row-major,
    // lanes=1 — the same beat order as io.c. Level contract: the S2 pass
    // controller holds passFirst/passLast stable from pass-fire to passDone.
    val spillIn = if (spill) Some(slave(Tensor(accType, Seq(M, N), lanes = 1))) else None
    val spillOut = if (spill) Some(master(Tensor(accType, Seq(M, N), lanes = 1))) else None
    val passFirst = if (spill) Some(in Bool()) else None
    val passLast = if (spill) Some(in Bool()) else None
    // Single-cycle pulse on every pass-drain completion (spill or final).
    val passDone = if (spill) Some(out Bool()) else None
  }
  
  // ==========================================
  // LOGARITHMIC ADDER TREE
  // ==========================================
  def buildAdderTree(inputs: Seq[TAcc], enable: Bool): TAcc = {
    if (inputs.length == 1) return inputs(0)
    
    val nextEnable = if (pipelineTree) RegNext(enable, init = False) else enable
    
    val nextStage = inputs.grouped(2).map { group =>
      if (group.length == 2) {
        val sum = (group(0), group(1)) match {
          case (a: SInt, b: SInt) => (a + b).resized.asInstanceOf[TAcc]
          case (a: UInt, b: UInt) => (a + b).resized.asInstanceOf[TAcc]
          case (a: spinalML.dtypes.FloatML, b: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(a, b).asInstanceOf[TAcc]
        }
        if (pipelineTree) RegNextWhen(sum, enable, init = sum.getZero) else sum
      } else {
        if (pipelineTree) RegNextWhen(group(0), enable, init = group(0).getZero) else group(0)
      }
    }.toSeq
    
    buildAdderTree(nextStage, nextEnable)
  }

  // Common accumulators (M rows, N cols)
  val accumulators = Vec(Reg(accType), M * N)
  accumulators.foreach(acc => acc.init(acc.getZero))
  
  def getAccIdx(idx: UInt): UInt = if (M * N == 1) U(0) else idx.resized
  
  val rowCounter = Counter(M)
  val nCounter = Counter(N)
  val kCounter = Counter(chunksK)
  val outCounter = Counter(M * N)

  io.a.stream.ready := False
  io.c.stream.valid := False
  io.c.stream.payload(0).assignFromBits(B(0, widthOf(accType) bits))
  // S1 spill-port defaults (no-ops when spill=false: the Options are empty).
  io.spillIn.foreach(_.stream.ready := False)
  io.spillOut.foreach(_.stream.valid := False)
  io.spillOut.foreach(_.stream.payload(0).assignFromBits(B(0, widthOf(accType) bits)))
  io.passDone.foreach(_ := False)

  if (parallelN) {
    // ==========================================
    // PARALLEL N ARCHITECTURE
    // ==========================================
    // B is buffered in N parallel StreamDoubleBuffers, each storing 1 column (K elements)
    val buffersB = Seq.fill(N)(StreamDoubleBuffer(dataType, paddedK, lanes))
    buffersB.foreach(_.io.reArm := io.reArm)
    
    val loadBankCounter = Counter(N)
    val loadElemCounter = Counter(chunksK)
    
    io.b.stream.ready := Vec(buffersB.map(_.io.streamIn.ready))(loadBankCounter.value)
    for (i <- 0 until N) {
      buffersB(i).io.streamIn.valid := io.b.stream.valid && (loadBankCounter.value === U(i))
      buffersB(i).io.streamIn.payload := io.b.stream.payload
      buffersB(i).io.readAddr := kCounter.value.resized
      buffersB(i).io.nextTile := False
    }
    
    when(io.b.stream.valid && io.b.stream.ready) {
      loadElemCounter.increment()
      when(loadElemCounter.willOverflowIfInc) {
        loadBankCounter.increment()
      }
    }
    
    val allTilesReady = buffersB.map(_.io.tileReady).reduce(_ && _)
    
    val stage1_fire = False
    val stage2_valid = RegNext(stage1_fire, init = False)
    val stage2_row = RegNextWhen(rowCounter.value, stage1_fire)
    val stage2_a = RegNextWhen(io.a.stream.payload, stage1_fire)
    
    val isLastChunk = kCounter.value === (chunksK - 1)
    
    val stage2_a_masked = Vec(dataType, lanes)
    for (i <- 0 until lanes) {
      val validLane = if (K % lanes == 0) True else !isLastChunk || U(i) < U(K % lanes)
      stage2_a_masked(i) := Mux(RegNextWhen(validLane, stage1_fire), stage2_a(i), stage2_a(i).getZero)
    }

    // N parallel multiplier arrays using universal DspMul (latency = 1)
    val multRegs = Seq.fill(N)(Vec(accType, lanes))
    for (n <- 0 until N) {
      for (i <- 0 until lanes) {
        multRegs(n)(i) := spinalML.dsp.DspMul(
          a = stage2_a_masked(i),
          b = buffersB(n).io.readData(i),
          enable = stage2_valid,
          accType = accType,
          latency = 1,
          dspConfig = dspConfig
        )
      }
    }
    
    // N parallel adder trees
    val stage3_enable = RegNext(stage2_valid, init = False)
    val tree_row = Delay(RegNextWhen(stage2_row, stage2_valid), treeLatency, when = True)
    val tree_valid = Delay(stage3_enable, treeLatency, init = False, when = True)
    
    val treeOutputs = for (n <- 0 until N) yield buildAdderTree(multRegs(n), stage3_enable)
    
    when(tree_valid) {
      for (n <- 0 until N) {
        val flatIdx = tree_row * N + n
        val currentAcc = accumulators(getAccIdx(flatIdx))
        val nextAcc = ((currentAcc, treeOutputs(n)) match {
          case (acc: SInt, sum: SInt) => (acc + sum).resized.asInstanceOf[TAcc]
          case (acc: UInt, sum: UInt) => (acc + sum).resized.asInstanceOf[TAcc]
          case (acc: spinalML.dtypes.FloatML, sum: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(acc, sum).asInstanceOf[TAcc]
        })
        accumulators(getAccIdx(flatIdx)) := nextAcc
      }
    }
    
    val fsm = new StateMachine {
      val stateWaitTile: State = new State with EntryPoint {
        whenIsActive {
          when(allTilesReady) {
            goto(stateComputeTile)
          }
        }
      }
      
      val stateComputeTile: State = new State {
        whenIsActive {
          io.a.stream.ready := True
          when(io.a.stream.valid) {
            stage1_fire := True
            kCounter.increment()
            when(kCounter.willOverflowIfInc) {
              rowCounter.increment()
              when(rowCounter.willOverflowIfInc) {
                buffersB.foreach(_.io.nextTile := True)
                goto(stateWaitFlushEnd)
              }
            }
          }
        }
      }
      
      val stateWaitFlushEnd: State = new State {
        val waitCounter = Counter(3 + treeLatency)
        whenIsActive {
          waitCounter.increment()
          when(waitCounter.willOverflowIfInc) {
            goto(stateOutput)
          }
        }
      }
      
      val stateOutput: State = new State {
        whenIsActive {
          io.c.stream.valid := True
          io.c.stream.payload(0) := accumulators(getAccIdx(outCounter.value))
          
          when(io.c.stream.ready) {
            accumulators(getAccIdx(outCounter.value)) := accumulators(0).getZero
            outCounter.increment()
            when(outCounter.willOverflowIfInc) {
               goto(stateWaitTile)
            }
          }
        }
      }
    }
    
  } else {
    // ==========================================
    // SEQUENTIAL N ARCHITECTURE
    // ==========================================
    // B is buffered in 1 StreamDoubleBuffer holding all N columns (size = paddedK * N)
    val bufferB = StreamDoubleBuffer(dataType, paddedK * N, lanes)
    bufferB.io.reArm := io.reArm
    bufferB.io.streamIn << io.b.stream
    bufferB.io.nextTile := False
    
    // We need to locally store a row of A (size paddedK) to reuse it N times.
    val memA = Mem(Vec(dataType, lanes), chunksK)
    val loadACounter = Counter(chunksK)
    val computeACounter = Counter(chunksK)
    
    bufferB.io.readAddr := (nCounter.value * chunksK + computeACounter.value).resized
    
    val stage1_fire = False
    val stage2_valid = RegNext(stage1_fire, init = False)
    val stage2_row = RegNextWhen(rowCounter.value, stage1_fire)
    val stage2_n = RegNextWhen(nCounter.value, stage1_fire)
    
    val readA = memA.readSync(computeACounter.value)
    
    val isLastChunk = RegNextWhen(computeACounter.value === (chunksK - 1), stage1_fire)
    val stage2_a_masked = Vec(dataType, lanes)
    for (i <- 0 until lanes) {
      val validLane = if (K % lanes == 0) True else !isLastChunk || U(i) < U(K % lanes)
      stage2_a_masked(i) := Mux(validLane, readA(i), readA(i).getZero)
    }
    
    val multRegs = Vec(accType, lanes)
    for (i <- 0 until lanes) {
      multRegs(i) := spinalML.dsp.DspMul(
        a = stage2_a_masked(i),
        b = bufferB.io.readData(i),
        enable = stage2_valid,
        accType = accType,
        latency = 1,
        dspConfig = dspConfig
      )
    }
    
    val stage3_enable = RegNext(stage2_valid, init = False)
    val tree_row = Delay(RegNextWhen(stage2_row, stage2_valid), treeLatency, when = True)
    val tree_n = Delay(RegNextWhen(stage2_n, stage2_valid), treeLatency, when = True)
    val tree_valid = Delay(stage3_enable, treeLatency, init = False, when = True)
    
    val treeOutput = buildAdderTree(multRegs, stage3_enable)
    
    // Accumulator table + slot selection. Legacy (temporal = 0): the whole
    // MxN partial table is registered, indexed by (row, col). Windowed
    // (temporal > 0): min(temporal, M) row-slots circularly indexed — each
    // row is drained the moment it completes, so the table + index mux
    // shrink. The sum order (row, col, chunk) is IDENTICAL in both modes.
    val (accTable, accIdxSel): (Vec[TAcc], (UInt, UInt) => UInt) = if (temporal >= 1) {
      val slots = M.min(temporal)
      val wt = Vec(Reg(accType), N * slots)
      wt.foreach(acc => acc.init(acc.getZero))
      val slotBits = scala.math.max(1, log2Up(slots + 1))
      val nBits = scala.math.max(1, log2Up(N + 1))
      val wIdx = (r: UInt, n: UInt) =>
        ((r % U(slots, slotBits bits)) * U(N) + n.resize(nBits)).resized
      (wt, wIdx)
    } else {
      (accumulators, (r: UInt, n: UInt) => getAccIdx(r * N + n))
    }

    when(tree_valid) {
      val flatIdx = accIdxSel(tree_row, tree_n)
      val currentAcc = accTable(flatIdx)
      val nextAcc = ((currentAcc, treeOutput) match {
        case (acc: SInt, sum: SInt) => (acc + sum).resized.asInstanceOf[TAcc]
        case (acc: UInt, sum: UInt) => (acc + sum).resized.asInstanceOf[TAcc]
        case (acc: spinalML.dtypes.FloatML, sum: spinalML.dtypes.FloatML) => spinalML.utils.Float.add(acc, sum).asInstanceOf[TAcc]
      })
      accTable(flatIdx) := nextAcc
    }
    
    // S1 spill slice-engine FSM: the windowed-drain FSM above, plus per-row
    // seeding (non-first passes) and a drain-target mux on exit (spill
    // region vs layer output). Seeding is row-interleaved — each row is
    // seeded just before its LoadA — because the window holds at most
    // `temporal` rows: seeding the whole MxN table upfront would overwrite
    // row r with row r+slots in the same slots. Drain order (row-major) and
    // seed order match, and within a pass each row's seed precedes its drain
    // (read-before-write on the same DDR row region: correct RMW order).
    // Counter invariant: every pass seeds exactly M rows (non-first) or zero
    // rows (first), so seedRow is 0 at every pass entry from reset on.
    // A pass is otherwise indistinguishable from a legacy command (reArm +
    // tileReady + row loop unchanged), so the S2 controller sequences passes
    // with the same signals as commands, with one hardening difference: pass
    // entry waits for an explicit pass-fire (rising edge of reArm), NOT for
    // a bare tileReady level. A stale-high tileReady left over from the
    // previous pass would otherwise self-trigger a phantom pass on old
    // passFirst/passLast levels during the controller's inter-pass latency.
    // Contract: one 1-cycle reArm pulse per pass (it also clears the B
    // buffer's tileReady, so the following tileReady wait always observes
    // the fresh slice fetch).
    val fsm = if (spill) {
      val rowBits = scala.math.max(1, log2Up(M + 1))
      val nBits = scala.math.max(1, log2Up(N + 1))
      val emitIdx = Reg(UInt(rowBits bits)) init (U(0))
      val emitLast = Reg(Bool) init (False)
      val emitCounter = Counter(N)
      val seedRow = Counter(M)
      val seedCol = Counter(N)
      // Seed-region pad drain (spillPadElems closes the last AXI beat).
      // Always elaborated (harmless when spillPadElems == 0: the emit path
      // below never enters the drain state, so no behavior change).
      val drainCounter = Counter(spillPadElems max 1)
      val prevReArm = RegInit(False)
      prevReArm := io.reArm

      new StateMachine {
        val stateWaitPass: State = new State with EntryPoint {
          whenIsActive {
            when(io.reArm && !prevReArm) {
              goto(stateWaitTile)
            }
          }
        }

        val stateWaitTile: State = new State {
          whenIsActive {
            when(bufferB.io.tileReady) {
              when(io.passFirst.get) {
                goto(stateLoadA)
              } otherwise {
                goto(stateSeedRow)
              }
            }
          }
        }

        // Seed one accumulator row from the previous pass's DDR partials
        // (N beats, same row-major order as the drain below). Entered once
        // per row on non-first passes: rows beyond the window would alias,
        // so each row is seeded just before its own LoadA. seedRow advances
        // on every visit (M visits per pass keep the 0-at-entry invariant).
        val stateSeedRow: State = new State {
          whenIsActive {
            io.spillIn.get.stream.ready := True
            when(io.spillIn.get.stream.valid) {
              accTable(accIdxSel(seedRow.value, seedCol.value)) := io.spillIn.get.stream.payload(0)
              seedCol.increment()
              when(seedCol.willOverflowIfInc) {
                seedRow.increment()
                goto(stateLoadA)
              }
            }
          }
        }

        val stateLoadA: State = new State {
          whenIsActive {
            io.a.stream.ready := True
            when(io.a.stream.valid) {
              memA.write(loadACounter.value, io.a.stream.payload)
              loadACounter.increment()
              when(loadACounter.willOverflowIfInc) {
                goto(stateComputeN)
              }
            }
          }
        }

        val stateComputeN: State = new State {
          whenIsActive {
            stage1_fire := True
            computeACounter.increment()
            when(computeACounter.willOverflowIfInc) {
              nCounter.increment()
              when(nCounter.willOverflowIfInc) {
                emitIdx := rowCounter.value.resize(rowBits)
                emitLast := rowCounter.value.resize(rowBits) === U(M - 1, rowBits bits)
                rowCounter.increment()
                when(rowCounter.willOverflowIfInc) {
                  bufferB.io.nextTile := True
                }
                goto(stateWaitFlush)
              }
            }
          }
        }

        val stateWaitFlush: State = new State {
          val waitCounter = Counter(3 + treeLatency)
          whenIsActive {
            waitCounter.increment()
            when(waitCounter.willOverflowIfInc) {
              goto(stateEmitRow)
            }
          }
        }

        val stateEmitRow: State = new State {
          whenIsActive {
            val emitFire = Bool()
            emitFire := False
            when(io.passLast.get) {
              io.c.stream.valid := True
              io.c.stream.payload(0) := accTable(accIdxSel(emitIdx, emitCounter.value.resize(nBits)))
              emitFire := io.c.stream.ready
            } otherwise {
              io.spillOut.get.stream.valid := True
              io.spillOut.get.stream.payload(0) := accTable(accIdxSel(emitIdx, emitCounter.value.resize(nBits)))
              emitFire := io.spillOut.get.stream.ready
            }
            when(emitFire) {
              val idx = accIdxSel(emitIdx, emitCounter.value.resize(nBits))
              accTable(idx) := accTable(idx).getZero
              emitCounter.increment()
              when(emitCounter.willOverflowIfInc) {
                when(emitLast) {
                  // Pass-drain completion: exactly one pulse per pass (not
                  // per row — emitCounter overflows on every row). On
                  // non-first passes with a non-beat-exact seed region, the
                  // last beat's pad elements are still parked in the seed
                  // reader's gearbox: drop-drain them first (passDone only
                  // afterwards), otherwise the next pass's first seed chunk
                  // wedges on the reader's accept gate or seeds row 0 with
                  // pad garbage. First passes fetch no seed: nothing to
                  // drain, pulse passDone immediately.
                  emitLast := False
                  if (spillPadElems > 0) {
                    when(io.passFirst.get) {
                      io.passDone.get := True
                      goto(stateWaitPass)
                    } otherwise {
                      drainCounter.clear()
                      goto(stateDrainPad)
                    }
                  } else {
                    io.passDone.get := True
                    goto(stateWaitPass)
                  }
                } otherwise {
                  // Next row: re-seed on non-first passes (window slots must
                  // be reloaded — the just-drained row was cleared); first
                  // passes stream on (drain-clear left zeros behind).
                  when(io.passFirst.get) {
                    goto(stateLoadA)
                  } otherwise {
                    goto(stateSeedRow)
                  }
                }
              }
            }
          }
        }

        // Seed-region pad drop-drain (entered from stateEmitRow on non-first
        // passes when spillPadElems > 0; unreachable otherwise). Consumes
        // exactly spillPadElems elements from spillIn and discards them, so
        // the seed reader's gearbox is empty when passDone pulses — the next
        // pass's first seed chunk is then accepted immediately and seeds
        // from valid data. Backpressure-only wait: the pad beats are already
        // commanded (or in flight behind consumed image beats), so they
        // always arrive; no bus dependency, no deadlock.
        val stateDrainPad: State = new State {
          whenIsActive {
            io.spillIn.get.stream.ready := True
            when(io.spillIn.get.stream.fire) {
              drainCounter.increment()
              when(drainCounter.willOverflowIfInc) {
                io.passDone.get := True
                goto(stateWaitPass)
              }
            }
          }
        }
      }
    } else if (temporal >= 1) {
      // Windowed-drain FSM: after each row's last product beats, the tree
      // pipeline (3 + treeLatency) is flushed, then the completed row is
      // emitted N beats to the output; only then is the next row loaded.
      val rowBits = scala.math.max(1, log2Up(M + 1))
      val nBits = scala.math.max(1, log2Up(N + 1))
      val emitIdx = Reg(UInt(rowBits bits)) init (U(0))
      val emitLast = Reg(Bool) init (False)
      val emitCounter = Counter(N)

      new StateMachine {
        val stateWaitTile: State = new State with EntryPoint {
          whenIsActive {
            when(bufferB.io.tileReady) {
              goto(stateLoadA)
            }
          }
        }

        val stateLoadA: State = new State {
          whenIsActive {
            io.a.stream.ready := True
            when(io.a.stream.valid) {
              memA.write(loadACounter.value, io.a.stream.payload)
              loadACounter.increment()
              when(loadACounter.willOverflowIfInc) {
                goto(stateComputeN)
              }
            }
          }
        }

        val stateComputeN: State = new State {
          whenIsActive {
            stage1_fire := True
            computeACounter.increment()
            when(computeACounter.willOverflowIfInc) {
              nCounter.increment()
              when(nCounter.willOverflowIfInc) {
                emitIdx := rowCounter.value.resize(rowBits)
                emitLast := rowCounter.value.resize(rowBits) === U(M - 1, rowBits bits)
                rowCounter.increment()
                when(rowCounter.willOverflowIfInc) {
                  bufferB.io.nextTile := True
                }
                goto(stateWaitFlush)
              }
            }
          }
        }

        val stateWaitFlush: State = new State {
          val waitCounter = Counter(3 + treeLatency)
          whenIsActive {
            waitCounter.increment()
            when(waitCounter.willOverflowIfInc) {
              goto(stateEmitRow)
            }
          }
        }

        val stateEmitRow: State = new State {
          whenIsActive {
            io.c.stream.valid := True
            io.c.stream.payload(0) := accTable(accIdxSel(emitIdx, emitCounter.value.resize(nBits)))
            when(io.c.stream.ready) {
              val idx = accIdxSel(emitIdx, emitCounter.value.resize(nBits))
              accTable(idx) := accTable(idx).getZero
              emitCounter.increment()
              when(emitCounter.willOverflowIfInc) {
                when(emitLast) {
                  emitLast := False
                  goto(stateWaitTile)
                } otherwise {
                  goto(stateLoadA)
                }
              }
            }
          }
        }
      }
    } else new StateMachine {
      val stateWaitTile: State = new State with EntryPoint {
        whenIsActive {
          when(bufferB.io.tileReady) {
            goto(stateLoadA)
          }
        }
      }
      
      val stateLoadA: State = new State {
        whenIsActive {
          io.a.stream.ready := True
          when(io.a.stream.valid) {
            memA.write(loadACounter.value, io.a.stream.payload)
            loadACounter.increment()
            when(loadACounter.willOverflowIfInc) {
              goto(stateComputeN)
            }
          }
        }
      }
      
      val stateComputeN: State = new State {
        whenIsActive {
          stage1_fire := True
          computeACounter.increment()
          when(computeACounter.willOverflowIfInc) {
            nCounter.increment()
            when(nCounter.willOverflowIfInc) {
              rowCounter.increment()
              when(rowCounter.willOverflowIfInc) {
                bufferB.io.nextTile := True
                goto(stateWaitFlushEnd)
              } otherwise {
                goto(stateWaitFlushA)
              }
            }
          }
        }
      }
      
      val stateWaitFlushA: State = new State {
        val waitCounter = Counter(3 + treeLatency)
        whenIsActive {
          waitCounter.increment()
          when(waitCounter.willOverflowIfInc) {
            goto(stateLoadA)
          }
        }
      }
      
      val stateWaitFlushEnd: State = new State {
        val waitCounter = Counter(3 + treeLatency)
        whenIsActive {
          waitCounter.increment()
          when(waitCounter.willOverflowIfInc) {
            goto(stateOutput)
          }
        }
      }
      
      val stateOutput: State = new State {
        whenIsActive {
          io.c.stream.valid := True
          io.c.stream.payload(0) := accumulators(getAccIdx(outCounter.value))
          
          when(io.c.stream.ready) {
            accumulators(getAccIdx(outCounter.value)) := accumulators(0).getZero
            outCounter.increment()
            when(outCounter.willOverflowIfInc) {
               goto(stateWaitTile)
            }
          }
        }
      }
    }
  }
}

object matmul {
  def apply[T <: Data, TAcc <: Data](
    a: Tensor[T],
    b: Tensor[T],
    accType: HardType[TAcc],
    parallelN: Boolean = false,
    reArm: Option[Bool] = None,
    temporal: Int = 0,
    dspConfig: spinalML.dsp.DspConfig = spinalML.dsp.DspConfig.default,
    // S1 spill threading (all None = legacy one-shot GEMM): the caller owns
    // the pass loop and provides the spill streams per pass.
    spill: Boolean = false,
    passFirst: Option[Bool] = None,
    passLast: Option[Bool] = None,
    spillSource: Option[Tensor[TAcc]] = None,
    spillSink: Option[Tensor[TAcc]] = None,
    passDone: Option[Bool] = None,
    // Trailing pad elements closing the seed region's last AXI beat (see
    // MatmulOp spillPadElems). 0 = beat-exact region (legacy behavior).
    spillPadElems: Int = 0
  ): Tensor[TAcc] = {
    val rankA = a.shape.length
    val rankB = b.shape.length
    require(rankA >= 2 && rankB >= 2, "Matmul requires at least 2D tensors")

    val M = a.shape(rankA - 2)
    val K_A = a.shape(rankA - 1)
    val K_B = b.shape(rankB - 2)
    val N = b.shape(rankB - 1)

    require(K_A == K_B, s"Inner dimensions must match (A.cols=$K_A == B.rows=$K_B)")
    require(a.lanes == b.lanes, s"Tensors must have the same lanes (${a.lanes} != ${b.lanes})")

    val batchDimsA = a.shape.dropRight(2)
    val batchDimsB = b.shape.dropRight(2)

    // For now, require batch dimensions to match exactly for Batched Matmul.
    // E.g., [Heads, SeqLen, Dim] x [Heads, Dim, SeqLen].
    require(batchDimsA == batchDimsB, s"Batch dimensions must match ($batchDimsA != $batchDimsB). Broadcasting stream B is not supported natively here.")

    val outShape = batchDimsA ++ Seq(M, N)

    val matmulComp = MatmulOp(a.dataType, accType, Seq(M, K_A), Seq(K_B, N), a.lanes, parallelN = parallelN, temporal = temporal, dspConfig = dspConfig, spill = spill, spillPadElems = spillPadElems)
    matmulComp.io.reArm := reArm.getOrElse(False)
    if (spill) {
      matmulComp.io.passFirst.get := passFirst.getOrElse(False)
      matmulComp.io.passLast.get := passLast.getOrElse(True)
      passDone.foreach(_ := matmulComp.io.passDone.get)
      spillSource.foreach(s => matmulComp.io.spillIn.get.stream << s.stream)
      spillSink.foreach(s => s.stream << matmulComp.io.spillOut.get.stream)
    }
    
    // Connect the continuous batched streams directly to the 2D MatmulOp.
    // MatmulOp natively loops back to stateWaitTile after each 2D matrix, allowing zero-overhead batching.
    matmulComp.io.a.stream << a.stream
    matmulComp.io.b.stream << b.stream
    
    // Reconstruct a Tensor with the proper 3D/4D shape
    val outTensor = Tensor(accType, outShape, 1)
    outTensor.stream << matmulComp.io.c.stream
    outTensor
  }

  def apply[T <: Data](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    apply(a, b, a.dataType)
  }
  
  def apply[T <: Data](a: Tensor[T], b: Tensor[T], parallelN: Boolean): Tensor[T] = {
    apply(a, b, a.dataType, parallelN = parallelN)
  }
}
