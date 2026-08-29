# W4A8 Prefetch-Overlap Corruption — Session Notes (Aug 2026)

**Scope**: `WeightPrefetchChainTest` red on `eager#0` (W4A8: full logit
mix — later BF16 too) and on `serial#1` / the last logits. The harness
asserts that EAGER reloads (RESIDENT+PREFETCH_EN, CSR 0x10=3 + RELOAD)
fetch the weight region during the IDLE window (zero weight ARs visible in
the START→first-beat window) while every pass stays bit-exact against
`Mnistw4a8Replica.logitsK(_, 4)`.

**Status**: ROOT CAUSE #1 (the eager total mix) **FOUND & FIXED** — the
eager pass is bit-exact on both models (max|hw-sw| = 0.000, 0 overlap ARs).
**REMAINING**: the *last*-logit residue (`serial#1`: 2 last logits;
`eager#0`: logit[9] off by its bias value, e.g. +0.25 in W4A8): the
`BiasAddOp` bias cache has no command boundary (see §4).

---

## 1. What was ruled out (evidence chain)

Everything below was tested in isolation and came back GREEN, one by one:

1. **`SdbSwapTb`** (test file `SdbSwapTb.scala`): `StreamDoubleBuffer` +
   `DoubleBufferStreamer` alone, EXACT W4A8 geometry (depth 2880, 4 lanes,
   720 beats), governed overlap + flip. Pass0 AND pass1 (post-flip) are
   bit-perfect. The swap machine itself is sound.
2. **`DmaSdbTb`** (same file): the full DDR path
   `DMAReader(wType, (288,10), outLanes=4, trimToElements, flushableGearbox)`
   → `StreamDoubleBuffer` → `DoubleBufferStreamer` with a real
   `AxiMemorySim`. 0/2880 elements differ. The fetch/repack chain is sound.
3. The eager failure is **reproducible bit-identically** across runs with
   the same sim seed, but changes between Verilator seeds → the corruption
   depends on **uninitialised FIFO/RAM cells** (Verilator random init:
   observed garbage patterns 0xBB/0x79 at t=0). Classic stale-state, not a
   timing race.

## 2. Root cause #1 — streamer delivery FIFO never flushed in prefetch world

`DoubleBufferStreamer` keeps a 16-deep `StreamFifo` between the BRAM read
and `streamOut` (absorbs the 1-cycle `readSync` latency). At the end of a
pass the Linear leaves up to 2-3 beats in flight in that FIFO.

In `Sequential.scala` the re-arm of the streamer was:

```scala
wStreamer.io.reArm := reqW.fire && !prefetchWorldW
```

The `!prefetchWorldW` guard was introduced for the *bank* re-arm (holding
resident banks must not reset `StreamDoubleBuffer`), but the SAME signal
carries the FIFO flush of the streamer. In the prefetch world the FIFO
therefore never gets purged: the next pass starts by draining 2-3 stale
beats of the PREVIOUS weight generation (content from an un-initialised
bank — seed-dependent), shifting the whole Linear by the tail and mixing
the last outputs.

Confirming trace (spy harness): the eager reload `FIRE` happens while the
streamer is STILL reading (`isReading` high, e.g. `readCycles=8794` for a
single 720-beat tile — the Linear dribbles at ~1 beat/12 cycles when
backpressured). The stale FIFO tail is then pushed first into the new pass.

**FIX** (3 lines, `spinalML/src/spinalML/nn/Sequential.scala`, kept):

```scala
wStreamer.io.reArm      := reqW.fire          // flush delivery FIFO every fetch
wDoubleBuffer.io.reArm  := reqW.fire && !prefetchWorldW   // banks still protected
...
bDoubleBuffer.io.reArm  := reqB.fire
bStreamer.io.reArm      := reqB.fire
```

Verification: `WeightPrefetchChainTest` p=1: baseline + serial#0 exact;
`eager#0` **bit-exact** (0.000 / predicted 1, label 1) on W4A8 **and**
BF16; weight ARs in the START→first-beat window = 0 (overlap proven).
Non-prefetch modes are algebraically unaffected (`&& !prefetchWorldW` was
already `true` there).

## 3. What the spy trace showed (full eager state machine)

Instrumented events per pass (last weight layer):

```
FIRE addr=0x00040028 staged=1 reload=0 ac=1 [sb=0 cb=0 pf=1 qf=0 lb=1]
... NEXT-TILE ... READ-END cycles=720 ... FLIP (cb 0→1, pf 1→0) ...
FILLED (idle-fill, qf=1) ... SETTLED ... (next pass)
```

Also visible: the matmul-based Linear consumes its weight streamer in
MULTIPLE tiles per pass (2-3 × 720 beats — per-N re-runs of the K axis),
so "one nextTile" ≠ "end of pass": a governed flip keyed on `nextTile`
still lands INSIDE a pass. (The bank swap itself is correct — see §1 —
but it must NOT be the solution for the boundary: `stageRequest` forced to
False leaves the failure unchanged.)

## 4. Remaining: the bias boundary (`BiasAddOp`), 2 last logits

Once the FIFO is flushed, `eager#0` is 9/10 exact and `serial#1` shows
only the last 2 logits wrong, off by the bias value of a previous
generation (W4A8: `+0.25` on logit[9] == fp8 bias residue; `+0.5/+0.25`
on 8/9 in `serial#1`).

Cause: the bias path has NO boundary at all:
- `BiasAddOp` (`ops/bias_add.scala`) loads its N biases into a `Vec(Reg)`
  cache (`stateLoadBias`); an in-flight reload (b-streamer re-arm) kills
  the old read mid-load (spy: `B-READEND cycles=2` at the instant of the
  eager `B-FIRE`), and the cache ends up with a half-old mixture that
  never gets cleared.
- `MatmulOp`'s internal `bufferB` (`StreamDoubleBuffer(paddedK*N)`) is
  re-armed via `weightDmaFire` but the BiasAddOp is NOT.
- The dequant cast path (`cast(io.w, ...)` for FP8→float) has no flush
  either — same after-boundary, low-priority suspect.

Attempted-ahead fail: giving `BiasAddOp` a `reArm` wired to the *weight*
fire (`weightDmaFire`) made things WORSE (6/10 wrong) — the correct
trigger for the bias cache is the **`reqB.fire`** (bias region cmd fire),
NOT the weight fire. Plumb it through `LinearLayer` (add a
`biasReArm: Option[Bool]`). Do that, re-run §7, then `serial#1` should
close too.

## 5. Residual items / low-priority suspects

- `repack.op` for the Linear inputs uses `withFlush=false` (M1.7 guard):
  stateless after end-of-group, fine for serial#, but validate only if
  leftovers persist after §4.
- Double-buffer `dataBank` RAM grows `pingFull/pongFull` with no
  counter-part for "empty": the second bank always carries the previous
  generation content — invisible today because baseA/baseB copies are
  byte-identical, but a REAL change of content would need the swap logic
  done first (see `stageRequest`/`refreshSettled` in `StreamDoubleBuffer`).

## 6. Files touched / relevant files

- `spinalML/src/spinalML/nn/Sequential.scala` — THE fix (3 lines, §2).
- `spinalML/src/spinalML/memory/DoubleBufferStreamer.scala` /
  `StreamDoubleBuffer.scala` — untouched now (instrumentation removed),
  re-read them for the §4 fix.
- `spinalML/src/spinalML/ops/bias_add.scala` — NOT fixed (bad experiment
  reverted): the §4 surfacing point.
- `spinalML/test/src/spinalML/examples/WeightPrefetchChainTest.scala` —
  untouched, oracle `logitsK(_, 4)`.
- `spinalML/test/src/spinalML/examples/SdbSwapTb.scala` — the two green
  micro-benchmarks (SdbSwapTb + DmaSdbTb), committed.
- `spinalML/src/spinalML/ops/matmul.scala` — `bufferB` internals,
  `parallelN=false` path.

## 7. Repro / verification commands

```bash
# Single eager pair, W4A8 only (fast repro of the remaining bias bug)
MNIST_PREFETCH_PAIRS=1 W4A8_ONLY=1 timeout 900 ./mill spinalML.test.testOnly spinalML.examples.WeightPrefetchChainTest
# Both models, 2 pairs (full)
timeout 1200 ./mill spinalML.test.testOnly spinalML.examples.WeightPrefetchChainTest
# Resident chain (must stay green)
timeout 900 ./mill spinalML.test.testOnly spinalML.examples.WeightResidentChainTest
# Micro-benchmarks
timeout 900 ./mill spinalML.test.testOnly spinalML.examples.SdbSwapTb spinalML.examples.DmaSdbTb
```

---

# Appendix — How to instrument probes effectively (for the next agent)

## Why this doc exists

This session burned a LOT of cycles on SpinalHDL probe plumbing. The
recipes below are what actually worked, the traps are what actually bit.

## The working pattern (copy this)

1. **Declare a probe Bundle** in the RTL file (package level):

```scala
class WeightProbe extends Bundle {
  val reqValid = Bool()
  val reqFire  = Bool()          // give a name; simulate via edges
  val staged   = Bool()
  val computeBank = Bool()
  ...
}
```

2. **Expose ONE out port** on the component's `io`:

```scala
val weightProbe = out(new WeightProbe)   // Sequential-level tap
```

3. **Wire the fields + make them sim-visible** (BOTH steps mandatory):

```scala
// ALSO: import spinal.core.sim._ at the top of the FILE — otherwise the
// `.simPublic()` extension is unresolved.
weightProbeSig.reqValid := reqW.valid
weightProbeSig.reqValid.setName("wProbe_reqValid")
weightProbeSig.reqValid.simPublic()
```

At the end of the component, ALSO publicise the module port itself
(the child-module boundary otherwise shadows the access):

```scala
io.weightProbe.elements.foreach(_ ._2.simPublic())
```

4. **Place the connect statements LAST** in the component body. If a probe
   is wired before the `RegInit`ed signals exist, elaboration errors with
   `:= null` (`Assignment data type mismatch ... := null`).

5. **In the test**: attach `onSamplings` with edge latches and a Scala
   cycle counter (there is NO `getSimTime` on `ClockDomain` in this SpinalHDL):

```scala
var lastFire = false; var cyc = 0
dut.clockDomain.onSamplings {
  cyc += 1
  val p = dut.model.io.weightProbe
  if (p.reqFire.toBoolean && !lastFire)
    log.append(f"$cyc%06d FIRE staged=${if (p.staged.toBoolean) 1 else 0} ...\n")
  lastFire = p.reqFire.toBoolean
}
```

6. **In the DUT hierarchy**: the tap sits on `Sequential`; from a
   top-level sim use `dut.model.io.weightProbe` (model = the
   `Accelerator.model`).

## Traps that WILL bite

| Trap | Symptom | Fix |
|---|---|---|
| Probes on `Stream` fields (`reqStream.fire`, `.valid`) | `NullPointerException ... Stream.valid() ... is null` | expose ONLY derived hard signals; convert combined events via `RegNext` (e.g. `RegNext(reqW.fire) init(False)`) — one cycle skew is acceptable for tracing |
| `Counter.value` referencing | same NPE family (`Counter.value() is null`) | don't tap counters directly; reconstruct them in the TEST from the fire edges |
| Missing `setName`/`simPublic` | sim aborts at first sample: `UNACCESSIBLE SIGNAL : (...) isn't accessible ... call simPublic()` | add both, on the per-field AND the io port (`elements.foreach(_._2.simPublic())`) |
| Probe wired before regs | `Assignment data type mismatch := null` | move the connect block to the END of the component |
| Assigning a probe twice (in-loop `:=` then fallback `io.x := new Probe`) | `ASSIGNMENT OVERLAP` elaboration failure | single driver only; make the fallback Scala-conditional (if/else), or always-assign at the end |
| Probe port left open for weightless models | `require(lastWeightLayer >= 0)` — kills unrelated suites | NEVER add a hard require for a diagnostic port; default-construct the probe bundle without firing it |
| Editing the production `test` files | CI/QA noise | clone the harness into a NEW test file (e.g. `PrefetchSpyTb.scala`), delete it before commit |
| `import spinal.core.sim._` used in RTL | breaks nothing but pollutes | keep it, but delete the whole probe block before the PR |

## Short checklist for the next run

1. `git diff --stat` must show ONLY the intended production lines before
   committing (the instrumented session above ballooned to 153 lines).
2. Delete the spy test file (`rm .../PrefetchSpyTb.scala`) and all probe
   ports from RTL before finalising the PR.
3. For the bias fix: plumb a SEPARATE `biasReArm` into `LinearLayer` and
   drive it with `reqB.fire` (NOT `reqW.fire`).
