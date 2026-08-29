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
  temporal: Int = 0
) extends Component {
  val M = shapeA(0)
  val K = shapeA(1)
  val N = shapeB(1)
  require(shapeB(0) == K, "Inner dimensions must match (A.cols == B.rows)")
  require(temporal >= 0, s"temporal=$temporal must be >= 0")
  require(temporal == 0 || !parallelN,
    s"temporal=$temporal requires the sequential-N matmul (parallelN=false)")
  
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

    // N parallel multiplier arrays
    val multRegs = Seq.fill(N)(Vec(Reg(accType), lanes))
    for (n <- 0 until N) multRegs(n).foreach(r => r.init(r.getZero))
    
    when(stage2_valid) {
      for (n <- 0 until N) {
        for (i <- 0 until lanes) {
          multRegs(n)(i) := ((stage2_a_masked(i), buffersB(n).io.readData(i)) match {
            case (valA: SInt, valB: SInt) => (valA * valB).resized.asInstanceOf[TAcc]
            case (valA: UInt, valB: UInt) => (valA * valB).resized.asInstanceOf[TAcc]
            case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.mul(valA, valB).asInstanceOf[TAcc]
            case _ => throw new Exception("Type unsupported")
          })
        }
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
    
    val multRegs = Vec(Reg(accType), lanes)
    multRegs.foreach(r => r.init(r.getZero))
    
    when(stage2_valid) {
      for (i <- 0 until lanes) {
        multRegs(i) := ((stage2_a_masked(i), bufferB.io.readData(i)) match {
          case (valA: SInt, valB: SInt) => (valA * valB).resized.asInstanceOf[TAcc]
          case (valA: UInt, valB: UInt) => (valA * valB).resized.asInstanceOf[TAcc]
          case (valA: spinalML.dtypes.FloatML, valB: spinalML.dtypes.FloatML) => spinalML.utils.Float.mul(valA, valB).asInstanceOf[TAcc]
          case _ => throw new Exception("Type unsupported")
        })
      }
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
    
    val fsm = if (temporal >= 1) {
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
  def apply[T <: Data, TAcc <: Data](a: Tensor[T], b: Tensor[T], accType: HardType[TAcc], parallelN: Boolean = false, reArm: Option[Bool] = None, temporal: Int = 0): Tensor[TAcc] = {
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

    val matmulComp = MatmulOp(a.dataType, accType, Seq(M, K_A), Seq(K_B, N), a.lanes, parallelN = parallelN, temporal = temporal)
    matmulComp.io.reArm := reArm.getOrElse(False)
    
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
