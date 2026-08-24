# MNIST Bring-Up Session — August 2026

Summary of the first **real trained-network bring-up** through the high-level
API: the "Bébé Cadum" MNIST CNN (2 942 parameters, 96 % float32 accuracy)
compiled end-to-end to an `Accelerator` SoC and validated black-box under
Verilator — final result **5/5 digits correctly classified**, hardware logits
matching an independent BF16 software replica to within 0.0005.

Three framework defects were uncovered and fixed (_PATCHED_), one protocol
limitation lifted (_PATCHED_), one accuracy pitfall documented, and one
known limitation formally characterized. The new burst-capable DMAReader is
now proven by formal properties (BMC PASS).

---

## What was built

* [`spinalML/src/spinalML/examples/Mnist.scala`](../../spinalML/src/spinalML/examples/Mnist.scala)
  — `Accelerator(BF16(), Seq(28, 28, 1), Seq(Conv2D(1→2 K5), ReLU(),
  MaxPool2D(2,2), Flatten(), Linear(288→10)))` plus the hard-coded trained
  weights (`MnistWeights`) and five binarized digits (`MnistData`, labels
  `[7, 2, 1, 0, 4]`). Raw data files live in `Mnist/*.txt`.
* [`spinalML/test/src/spinalML/examples/MnistTest.scala`](../../spinalML/test/src/spinalML/examples/MnistTest.scala)
  — black-box SoC bench: DDR filled exactly like a real host, control
  registers programmed over AXI4-Lite, 10 output beats collected per digit,
  prediction = argmax. **No golden model**: the trained network is the oracle.
* `Mnist/gen_golden.py`-style replication (debug only, not kept): the existing
  `tests/python/golden_models/ops.py` goldens were used *diagnostically* to
  prove the final HW matches the intended BF16 dataflow bit-for-bit.

Two data adaptations were needed (framework conventions, not bugs):

1. **Features-last input**: PyTorch `[1, 28, 28]` becomes `[28, 28, 1]`.
2. **Flatten-order remap of the FC weights**: Torch flattens `[C,H,W]`
   (channel-major); spinalML `Flatten` is features-last `[H,W,C]`. Column
   *k* of each stored FC row must therefore be Torch column
   `c·144 + i·12 + j` where `k = (i·12 + j)·2 + c`. Without this remap the
   network computes perfectly valid — and perfectly wrong — logits.
3. The trained head is LogSoftmax over 10 classes; it is omitted because
   `Softmax1D` requires power-of-2 channels (adder tree) and
   `argmax(logits) = argmax(softmax(logits))`.

---

## Bug 1 — Requests longer than 255 beats were unexpressible — _PATCHED_

`FetchRequest.length` was `UInt(8 bits)` (raw AXI arlen width). The FC layer
alone needs 720 AXI beats (2 880 BF16 elements) → elaboration-time WIDTH
MISMATCH in `Sequential`.

### Fix
`length` widened to **16 bits** (`DMAReader.scala`), and the reader gained a
real burst engine:

* requests split into chained INCR bursts of at most `maxBurstBeats`
  (default 256) beats;
* every burst additionally clipped so it **never crosses a 4 KiB boundary**
  (AXI4 protocol rule);
* bursts are strictly serialized: the next AR is issued only after the
  previous burst fully drained, preserving element order trivially.

The 2D reader delegates its rows to the 1D engine, so both benefit.

## Bug 2 — Unaligned weight/bias regions silently read shifted data — _PATCHED_

The Sequential builder stacked weight regions back-to-back with byte
granularity. The conv bias landed at byte offset **100** — not 8-aligned.
Memory models (and real DDR controllers) serve a burst from the
beat-truncated address, so the hardware received elements 48/49 instead of
the bias: constant garbage offsets, predictions destroyed while everything
looked plausible. Symptom fingerprint: outputs constant per channel,
bias-like value equal to *some earlier weight bytes*.

### Fix
`Sequential` now aligns every weight/bias region start to the AXI beat
(`alignToBeat`, `Sequential.scala`). Benches must pad their packed blobs the
same way (see `MnistTest.weightWords`, `padded()` per section — padding the
concatenation instead of each section reproduces the bug).

## Bug 3 — Maximum-length request silently did nothing — _PATCHED_ (formal catch)

`remaining := cmd.length + 1` on a 16-bit counter wraps `0xFFFF + 1 → 0`:
a maximum-size request (65 536 beats) left the DMA permanently idle.
Found **by the new formal harness**, not by simulation.

### Fix
Beat counters widened to 17 bits; overflow-free `+^` addition.

## New formal proof — DMAReader burst engine

[`DMAReaderFormal.scala`](../../spinalML/test/src/spinalML/symbolicTest/memory/DMAReaderFormal.scala)
was rewritten around the burst semantics and proves, under AXI-legal
environment assumptions:

1. handshake structure against internal counters;
2. burst legality: `len+1 ≤ min(remaining, beats-to-4K-boundary)` — i.e. no
   burst ever crosses a 4 KiB edge;
3. chained-burst contiguity: split bursts continue at the exact INCR address;
4. **beat-counting integrity**: every command receives *exactly*
   `length+1` R beats — none dropped, none duplicated (this is the property
   class whose violation caused the Mnist deadlock below).

Verification speed trick: `DMAReader(maxBurstBeats = …)` lets benches shrink
the burst cap (formal uses 4) so multi-burst paths fit inside a handful of
BMC steps instead of 256+ cycles — `BMC(8)` suffices and runs in seconds.
Both `DMAReaderFormal` and `DMAReader2DFormal` report `DONE (PASS)`.

## Debug war stories worth remembering

* **SIGSEGV in Verilator on large designs**: the flat evaluation tree of a
  189 k-line design overflows the default 1 MB Java thread stack. Fix:
  `-Xss512m` on the test JVM (`build.sc`, `forkArgs`). Symptom was a bare
  exit-code 139 right after simulation start, independent of stimulus.
* **`bitVectorWidthMax`**: a 288-lane BF16 stream beat is 4 608 bits — above
  SpinalHDL's 4 096 sanity limit. Raise it per-simulation
  (`SpinalConfig(bitVectorWidthMax = 16384)`).
* **StreamWidthAdapter payload slicing trap**: `addrReg(11, 0 bits)` means
  *"start at bit 11, take 0 bits"* in SpinalHDL (`(offset, width)`), silently
  yielding a constant. The correct slice is `addrReg(0, 12 bits)`.
* **AR-overlap race**: letting the next burst's AR assert while the previous
  burst still drained made the `burstRemain` reload race its last decrements
  — one credit lost per burst frontier, tail beats masked out downstream,
  `tileReady` never asserted, pipeline deadlocked. Strict serialization
  fixed it; the formal contiguity/counting properties now pin it down.
* **Probe hygiene**: a debug harness that pads the concatenated weight blob
  instead of each section silently shifts every later region; several
  "impossible" observations during the hunt were self-inflicted. Pad
  per-section, always audit with `memory.readBigInt` read-backs.

## Known limitation confirmed — one-shot restarts

Running five inferences back-to-back inside a single `doSim` (rewrite image
base, pulse start again) corrupts every run after the first, while isolated
fresh simulations give 5/5. Some datapath state (double-buffer bank flags /
FSM counters) is not re-armed between starts. This matches the documented
one-shot contract (tutorial §8) and is now an explicit entry point for the
*Multi-Tile Continuous Inference* roadmap item. `MnistTest` sidesteps it by
giving each digit its own `doSim`.

---

## Validation status

| Suite | Result |
|---|---|
| Scala non-regression (all packages except symbolic) | 308 tests / 69 suites green |
| Python co-sim (dma_reader, dma_reader2d, double_buffer_streamer) | 3/3 |
| Formal `DMAReaderFormal` (burst engine, BMC(8)) | DONE (PASS) |
| Formal `DMAReader2DFormal` | DONE (PASS) |
| `spinalML.examples.MnistTest` black-box | 5/5 |

Per-inference latency observed in simulation: ≈ 9.5 k cycles for the whole
network (serial conv over 576 windows, lanes=1 activations).

---

## Improvement leads (upscaling)

1. **Restartable inference** — root-cause the residual state between `start`
   pulses (prime suspects: `StreamDoubleBuffer` bank-full flags and op FSM
   counters that only reset through their own `stateDone`). Prerequisite for
   the continuous-execution roadmap item; would also make benches able to
   loop datasets inside one simulation.
2. **Wider accumulators** — conv/matmul accumulate in the activation dtype
   (BF16 here). An FP32-class accumulator (`FloatML(8, 15+)`) or int32 for
   integer paths would buy accuracy headroom on deeper nets; needs a mid-chain
   `Cast` or a wide global dtype (DMA beat packing adapts automatically,
   halving `elementsPerBeat`).
3. **Throughput** — everything is lanes=1 on the activation path and the conv
   walks windows serially. Levers, cheapest first: weight double-buffer
   prefetch of layer *n+1* during layer *n* compute (pure scheduling),
   `parallelN` matmul mode, wider im2col lanes, then the multi-tile
   continuous execution model.
4. **Scale limits to lift for bigger models**: `FetchRequest` tops at 64K
   beats per tensor (chunked-fetch API or wider field), and conv line buffers
   assume the whole feature map streams (advanced tiling roadmap item).
5. **Tooling** — an ONNX→DDR-blob packer emitting the padded, reordered weight
   region (and the flatten-order remap) would remove the last manual step of
   the chain and pair naturally with the future GUI flow.
