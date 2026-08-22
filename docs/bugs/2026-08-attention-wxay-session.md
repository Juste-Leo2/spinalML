# Attention wXaY Debug Session — August 2026

Summary of the bug hunt that accompanied the weight-only quantization (`wXaY`)
bring-up of Classical/Multi-Head attention. Three RTL bugs were found, proven
and fixed (_PATCHED_), plus two test-harness fixes (_PATCHED_).

---

## Bug 0 — Float units silently truncated instead of rounding — _PATCHED_

Context for the two attention bugs below: their symptoms were only visible
because the activation math got **more precise**, which shifted every numeric
comparison. The root cause of the reduced precision lived in
`spinalML/src/spinalML/utils/Float.scala`.

### Root causes (two independent defects)
1. **Exponent wrap**: in both `Float.mul` and `Float.add` the intermediate
   exponent was computed on `expBits+1` signed bits without expansion. Beyond
   the representable max the exponent wrapped negative → output became **zero
   instead of saturating to ±inf** (e.g. FP4: `3.0 × 3.0 = 0`). Fix: widen the
   intermediates (`expSumWidth = expBits + 3`).
2. **Truncation instead of round-to-nearest**: `finalMantissa = normalized >> k`
   discarded guard/sticky bits. Classic absorption:
   `1.0 + 0.0074 → 1.0` instead of `1.0078125`, which broke softmax
   normalization downstream. Fix: **round-to-nearest-even**
   (`guard && (sticky || LSB)`), with the rounding carry propagated into the
   exponent.
   ⚠️ SpinalHDL trap hit on the way: `Bool.asSInt` evaluates to **−1** when
   true — use `.asUInt.intoSInt.resized` for carry injection.

### How it was caught / regression-proofed
Random sampling never saw the absorption corner. A systematic sweep did:
`FloatSweepTest.scala` now runs exhaustive FP4 (121 pairs) and FP8 (51 529
pairs) multiplication/addition tables plus a targeted **near-1.0 absorption
sweep** (~3.5M BF16 pairs over `add(a ∈ [1..4], small b)`) against an
independent oracle (`FloatGolden`, unbounded integer arithmetic).

All golden models were synchronized to the corrected semantics in the same
movement: `tests/python/golden_models/ops.py` (`floatml_mul`, `floatml_add`)
and the Scala `FloatGolden`. Bit-exactness between HW and goldens is what made
the later attention bugs measurable at all (trial-level diffs of 0.001 instead
of tolerance mush).

> Open follow-up: `test_cast.py`, `test_matmul.py`, `test_softmax.py` were last
> run *before* the rounding change; re-run them to confirm bit-exactness of
> those paths against the updated goldens.

---

## Bug 0bis — Context matmul consumed `Vᵀ`: `P @ Vᵀ` instead of `P @ V` — _PATCHED_

Same convention clash as Bug 1, opposite direction, fixed one session earlier:
the projection matmul streams `V` **row-major** while the context matmul
consumes its B operand **column-major**, so the raw stream is read as `Vᵀ`.
Fix in `ClassicalAttention.scala`: `transpose(v)` before the fifo/repack, with
the logical orientation re-declared as `[seqLen, headDim]` (mandatory in
multi-head, where the shape is non-square). Masked for months by relaxed
softmax tolerances and small random weights.

---

## Bug 1 — K matrix double-transposed: HW computed `softmax(Q·K)` — _PATCHED_

### Symptom
- wXaY quant attention cocotb tests failed from trial 1 onwards with large errors
  (e.g. `Y[0][0]: got 2.625 instead of 2.203125`, error 2.67 on some rows).
- **Seed-dependent**: seed 1337 passed all trials, seed 42 failed. This was NOT
  noise: it meant the defect was data-dependent, not a race.

### Evidence
Offline hypothesis hunt (numpy, replicating the exact seeded test data):
for every trial of both seeds the best match by far was
`softmax(Q·K) @ V @ Wo` (max diff **0.0015–0.027**), never the correct
`softmax(Q·Kᵀ) @ V @ Wo` (diff up to 2.7). Seed 1337 only "passed" because its
random K made `Q·K ≈ Q·Kᵀ` within the relaxed 0.25 tolerance — luck, not correctness.

An inverse-problem check (reconstructing the probs actually used by the HW via
pseudo-inverse of `V·Wo`) confirmed scores gaps matching neither golden row.

### Root cause
Two conventions cancel each other:

1. A matmul **output** stream carries rows of C (row-major).
2. A matmul **B-input** consumes columns of B (column-major weights).

Therefore feeding the raw `k` stream into a matmul B-input already yields `Q·Kᵀ`
(columns of `B = Kᵀ` are rows of `K`). The pre-existing explicit
`val k_t = transpose(k)` re-transposed an implicitly-transposed stream, so the
scores matmul received columns-of-K and computed `Q·K`.

This is the mirror image of the V bug fixed previously (there the transpose was
missing, here it was superfluous). Both were masked for months by relaxed
softmax tolerances and small random test data.

### Fix (`spinalML/src/spinalML/attention/ClassicalAttention.scala`)
Remove `transpose(k)`; wire the raw projection output through a queue only, and
declare the logical shape as `[headDim, seqLen]` (the B operand *is* `Kᵀ`
logically):

```scala
val k_fifo = Tensor(dataType, Seq(headDim, seqLen), 1)
k_fifo.stream << k.stream.queue(seqLen * headDim)
val q_fifo = Tensor(dataType, q.shape, q.lanes)
q_fifo.stream << q.stream.queue(seqLen * headDim)
val scores = matmul(q_fifo, k_fifo, dataType)
```

Note: the declaration shape matters — square single-head configs masked it
(`[seqLen, headDim] ≡ [headDim, seqLen]`), multi-head (`seqLen=4, headDim=2`)
surfaced it as an elaboration-time dimension mismatch until declared correctly.

### Validation
Classical + Multi-Head attention suites: quant combos `w8a16/w4a16/w8a8/w4a8`,
per-channel scales, uniform BF16/FP8 — all pass, most **bit-exact** vs golden
(e.g. classical w8a16 trial 1 abs_err = 0 everywhere).

---

## Bug 2 — Softmax1D final join replays a beat: duplicated output rows — _PATCHED_

### Symptom
Multi-head attention emitted `Y = [R0, R1, R1(bit-exact), R2]`; the last golden
row was lost. Happened inside trial 0, only under backpressure, only for some
seeds/shapes (seqLen=4). Single-head (seqLen=2) never hit the bad interleaving.

A bit-exact duplicated row means a payload register was presented twice — not a
math error.

### Evidence (VCD, non-intrusive)
Re-ran the failing cocotb test with Verilator tracing enabled and parsed the
waveform (see `docs/bugs/detect_bug.md` for the method). Beat streams:

```
H0 softmax.x : [S0, S1, S2, S3]              <- clean input
H0 softmax.y : [P0, P1, P1, P2]              <- duplicated beat, P3 never emitted
```

Posedge timeline of the final stage showed `io_y_stream_valid && ready` firing
twice with the same mantissa while the source pair `(invSum, carryExp)` had
already moved on — the output pipe register re-emitted stale content.

### Root cause
In `Softmax1D` the final invSum × carryExp join was gated by the **raw sink
ready** instead of the ready seen by the pipe input:

```scala
// BUGGY
invSumStream.ready := io.y.stream.ready && carryExpStream.valid
carryExpStream.ready := io.y.stream.ready && invSumStream.valid
...
io.y.stream << outStream.m2sPipe()
```

`m2sPipe` asserts `source.ready = !full || sink.ready`. When its register was
**empty**, it captured the join's beat even while `io.y.stream.ready = 0` — but
the join did not consume (its readies were still gated by `y.ready`). The same
pair stayed presented, and on the next `y.ready` pulse the pipe forwarded the
stored beat **and re-captured the identical pair**: one beat in, two beats out.

Every other join in the same file already used the correct pattern
(`currentStream.ready` comes from its own `m2sPipe`).

### Fix (`spinalML/src/spinalML/activations/softmax.scala`)
Gate the join with the pre-pipe stream's own ready:

```scala
val outStream = Stream(Vec(dataType, channels))
val finalSyncValid = invSumStream.valid && carryExpStream.valid
invSumStream.ready := outStream.ready && carryExpStream.valid
carryExpStream.ready := outStream.ready && invSumStream.valid
...
outStream.valid := finalSyncValid
outStream.payload := outPayload
io.y.stream << outStream.m2sPipe()
```

No data path, latency budget or math changed — handshake semantics only.

### Validation
- The exact failing scenario (`multihead_quant_w8a16`, seed 42) went from FAIL
  at trial 1 to PASS over 3 full trials.
- 19/22 attention cocotb tests green (the 3 remaining are the items below).
- Full Scala suite: 242/242.

---

## Bug 3 — Test asserts crash on legitimate `inf` saturations (FP4 activations) — _PATCHED_

FP4 activation combos (`wXa4`) legitimately saturate to `±inf` in BOTH hardware
and golden (dynamic range of E2M1). `abs(inf - inf) = nan`, so the tolerance
assert failed on agreeing values (`got inf instead of inf`).

Fix in both `tests/python/test_classicalattention.py` and
`tests/python/test_multiheadattention.py`: inf-aware comparison —

```python
if np.isinf(out_val) or np.isinf(exp_val):
    assert np.isinf(out_val) and np.isinf(exp_val) \
        and np.signbit(out_val) == np.signbit(exp_val), ...
else:
    assert err < 0.25, ...
```

After this, `w8a4`/`w4a4` pass on both attention suites.

---

## Also fixed in the same campaign
- **I16 (SInt) softmax polarity** — _PATCHED_: PWL reciprocal coefficient
  saturation made `recip(1) = -1` (hot probability inverted, Cosine −1.000 on
  the single-head I16 path). Piecewise-constant reciprocal for SInt > 8 bits;
  see `docs/bugs/int16-softmax-polarity.md`.
