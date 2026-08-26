# Inter-Start State Re-Arming — Session Notes (Aug 2026)

**Scope**: make back-to-back `start` pulses produce bit-exact repeated
inferences inside ONE simulation session (roadmap item *Multi-Tile
Continuous Inference / inter-start state re-arming*). This is the enabling
step for every later streaming feature (per-tile chaining, weight
residency, folding).

**Status**: DONE for the one-shot contract. MnistChainedTest runs 10
consecutive inferences per session on BOTH Mnist models (BF16 and W4A8),
each logit set bit-exact against the software replica
(`test/src/spinalML/examples/MnistReplica.scala`, port of
`utils/Float.scala`). Full Scala suite 50/50, Python DMA suites 3/3,
DagTopology 6/6.

---

## 1. Root causes (there were THREE, stacked)

### RC1 — Region padding pollutes exact-size double buffers
Weight/bias DDR regions are padded up to whole AXI beats (e.g. Conv2D bias
[2 x I16] = 4 B lives in an 8 B beat; FcB [10 x E4M3] in a 16 B beat). The
DMA therefore streams MORE elements than the logical tensor: the extra
"pad" beats are accepted by the downstream `StreamDoubleBuffer` into its
SECOND bank (it is not full), so at end of inference the buffer sits in a
state like `loadBank=Pong(counter=6), pingFull=1, pongFull=0`. On the next
`start` the streamer consumes whatever bank `computeBank` points at while
the fresh fetch lands elsewhere — activations/weights arrive shifted and
every inference after the first is corrupted.

Fingerprint from waveforms: `loadCounter_value = 6` at idle, logits that
look like half-stale/half-fresh mixtures.

### RC2 — Lane gearbox keeps a partial group across commands
Same regions, second victim: the lane adapter (e.g. 16 raw I4 elements per
beat repacked to 25-lane conv weights) delivers floor(N/ratio) complete
groups and PARKS the remainder (50 weights -> 2 groups of 25 + 14 elements)
in its internal state. The next command's bits append AFTER the stale
residue: the whole tile is phase-shifted by a constant offset. This one
corrupts even when the SAME image is re-run, because the WEIGHTS become
misaligned.

Fingerprint: identical-image chained run still diverges; gearbox counter
non-zero at idle.

### RC3 — No command boundary existed anywhere
Neither of the above is fixable from inside a module without KNOWING where
a new command begins. The Accelerator's `startPending` flag exists, but no
re-arm pulse was distributed to the datapath — that distribution is the
actual feature built in this session.

## 2. Boundary traps discovered on the way (each caused a deadlock)

1. **DMAReader2D asserts `cmd.ready` only on the LAST DRAINED BEAT**
   (`stateDrain`) — "late acceptance". So `dmaImg.io.cmd.fire` fires when
   the image has ALREADY filled a bank; pulsing a re-arm from it clears a
   fresh `tileReady` forever -> pipeline deadlock, 0 output beats.
2. **The synchronous StreamFork completes its handshake only when EVERY
   sink accepted** — so `io.start.fire` is bounded below by the slowest
   (the 2D image DMA) and lands at the same too-late moment. Deadlock #2.
3. **Correct early boundary**: rising edge of `io.start.valid`
   (`RegNext(init(False))` qualifier). It occurs one cycle after the host
   writes START, strictly before any DMA data moves.
4. **1D readers accept EARLY** (`cmd.ready` true at idle), so their own
   `reqW.fire` / `reqB.fire` are valid per-command boundaries for the
   weight/bias path — which is also exactly what Weight-Manager-era
   residency will want (a resident weight DMA simply stops firing, so its
   buffers stop being re-armed).
5. **SpinalHDL hierarchy**: a parent-scope `Bool` cannot be smuggled
   through child constructor parameters across TWO component levels
   (HIERARCHY VIOLATION). Signals crossing components must be real ports;
   object-level `apply()` helpers may connect child ports from the caller's
   scope legally.

## 3. Fixes shipped (final architecture)

| Fix | File | What |
|---|---|---|
| F1 | `memory/StreamDoubleBuffer.scala` | new `io.reArm` input: returns `loadBank/computeBank/pingFull/pongFull/loadCounter` to power-on state (last-assignment-wins). |
| F2 | `nn/Sequential.scala` | img buffer re-armed from **rising edge of io.start.valid**; weight/bias buffers re-armed from **their own `reqW/reqB.fire`** (residency-friendly). |
| F3 | `ops/matmul.scala`, `layers/{Conv1D,Conv2D,Linear}.scala` | `io.reArm` / optional `reArm: Option[Bool]` threading down to MatmulOp's internal B buffer(s); defaults keep every other call site byte-identical. |
| F4 | `memory/DMAReader.scala` | **`trimToElements`**: suppresses everything past `shape.product` emitted elements so weight/bias streams end GROUP-ALIGNED (kills RC1 at the source; counters restart at each cmd.fire). |
| F5 | `memory/DMAReader.scala` + `ops/repack.scala` | **`flushableGearbox`**: structured SPLIT/AGGREGATE gearbox with `io.reArm` flush + `io.isEmpty`; command acceptance is gated on `gearboxEmpty` so a flush can never truncate a draining tensor (kills RC2). Enabled ONLY on weight/bias readers. |
| F6 | `ops/repack.scala` | RepackOp becomes dual-mode: legacy SpinalHDL `StreamWidthAdapter` by default, structured flushable internals on demand (`withFlush = true`). |

Deliberately LEFT LEGACY: the image path (`DMAReader2D`'s internal 1D
reader keeps the plain width adapter). Image rows are exact multiples of
the lane ratio (row trim handles the beat tail), so no residue exists
there — and the structured gearbox currently breaks the Residual MLP DAG
test in a way that isolated micro-probes do NOT reproduce (see §5).

## 4. Debugging techniques that cracked it (kept for posterity)

- **Red test first**: MnistChainedTest reproduced corruption in one small
  sim before any fix existed.
- **Bit-exact JVM replica as oracle** (HWFloat): turned "looks wrong" into
  "element k differs by delta", and later made EVERY random vector a test.
- **Waveform state diffing**: snapshot ALL signals just before re-arm #1 vs
  #2 and print the differences — this surfaced both RC1
  (`streamDoubleBuffer_10.loadCounter_value=6`) and RC2
  (`widthAdapter_counter_value` residue) directly.
- **Fire-gated VCD extraction with posedge two-pass sampling**: naive
  payload tracing reads combinational wiggles between handshakes; sample
  only `valid && ready` at clock edges, apply all changes of a timestamp
  before sampling.
- **Micro-probes** (`RepackOp` standalone): caught three gearbox bugs in
  minutes that system sims obfuscated — always probe new stream components
  standalone with an eager consumer before integrating.

## 5. Open follow-ups

1. **Formal re-runs**: harnesses for StreamDoubleBuffer / MatmulOp /
   Conv1D-2D / Linear formals were updated for the new ports (`io.reArm`
   tied False — one-shot windows unchanged). ~~BMC proofs not yet
   re-executed.~~    **DONE — all formal suites pass in CI (GitHub), Aug 2026; confirmed
   locally 8/8 (CVC4 1.8, see `docs/open-mysteries.md` Annexe B).** New
   invariants
   worth adding: `cmd.fire ==> buffer state == power-on` per module;
   RepackOp bit-stream identity under arbitrary stalls is tracked as
   M1-étape B in `docs/open-mysteries.md`.
2. **Structured-gearbox × DAG latent bug**: with the flushable gearbox also
   driving the IMAGE path, `ResidualMLPTemplate` (skip connection) produces
   wrong second-row values even though standalone probes of the same
   configuration (4→1 split, BF16 lanes, SInt and FloatML) are bit-perfect.
   Suspect a pacing-sensitive assumption somewhere between im2col and the
   tap FIFO. Until understood, the image path intentionally stays on the
   legacy adapter. Dedicated register with ranked hypotheses, falsification
   experiments and the dissection plan: `docs/open-mysteries.md` (M1; im2col
   window state tracked there as M2).
3. **Chain tests with random inputs**: combine `MNIST_RANDOM_N` with
   chaining (`MNIST_CHAIN_N`) once convenient — the replica oracle makes
   every chained random vector a full-strength check.
4. **Soft-reset debug register** (optional, phase 4): a debug-only
   AXI-Lite register to recover a card after unknown-state bring-up; NOT
   part of the correction mechanism.

## 6. Validation summary

| Suite | Result |
|---|---|
| MnistChainedTest W4A8 N=10 | 10/10 bit-exact vs replica |
| MnistChainedTest BF16 N=10 | 10/10 bit-exact vs replica |
| MnistTest / Mnistw4a8Test (curated + MNIST_RANDOM_N=10) | bit-exact, labels OK |
| Full Scala suite (memory/nn/examples/test.*) | 50/50 |
| DagTopologyTest (DAG/skip connections) | 6/6 |
| Python DMA suites (dma_reader, dma_reader2d, double_buffer_streamer) | 3/3 |
