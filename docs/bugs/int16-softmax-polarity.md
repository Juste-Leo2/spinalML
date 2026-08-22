# I16 (SInt) Softmax Polarity Bug — OPEN, not patched

## Summary
The integer-domain softmax path (`SInt` activations wider than 8 bits, i.e.
`ExpOp`/`ReciprocalOp` PWL branches) produces a **sign-inverted "hot" probability**
and saturating garbage on later tiles. This is a **pre-existing** defect: it was
masked for months because I16 attention outputs are tiny and the tests use a
`%FS` tolerance.

## Symptoms
| Test | Observation |
|---|---|
| `cocotb_multihead_i16` (seed 42) | FAIL at `Y[1][1]: got 32699.0 instead of 69.0`, deterministic |
| `cocotb_attention_i16` (single-head, seed 42) | PASSES tolerance but reports **Cosine = −1.000**, MAE 10.0 — systematically anti-correlated with golden |

Cosine −1 in single-head proves a systematic polarity/semantic inversion, not
random corruption; the multi-head failure is the same defect amplified by larger
accumulated values.

## Evidence (VCD beat capture, seed 42)
Golden chain for trial 0 head 0 (offline numpy replication of the test data):

```
scores row0 = [0, -153]   -> golden probs = [1, 0]
scores row1 = [-42, 9]    -> golden probs = [0, 1]      # one-hot, +1
```

Hardware softmax output beats (4-wide rows):

```
t=1250  (0, 0, -1, 0)     <- hot lane is -1, expected +1
t=1330  (0, 0, 0, -1)
...
trial 1:
t=3480  (0, 32767, 32767, 0)   <- saturation explosion
concat: (-32749, -32749, -3, -12)
```

`I16.from_float(1.0) == 1` (no scaling ambiguity): golden one-hot is unambiguously
positive, so the hardware `-1` is a genuine sign bug, not an encoding convention.

## Suspected root cause (to be confirmed)
The SInt > 8-bit unary ops index their approximation LUTs with the **raw top
bits of the two's-complement value** (sign bit included), e.g.
`ops/exp.scala`:

```scala
val segmentIndexFn: T => UInt = (x: T) =>
  x.asBits(bitWidth - 1 downto bitWidth - indexBits).asUInt
```

Consequences to audit:
1. Negative inputs map to high segment indices whose stored values were built
   from positive abscissae — semantics of `exp(negative)` in that region are
   undefined-ish (should be ~0).
2. `ReciprocalOp`'s `mathFn = 1/(x + 1e-9)` is applied per-segment over raw-bit
   abscissae; for negative sums this yields large negatives, and encode
   saturation can flip the hot lane to −1.
3. The golden model (`pwl_int`) replicates the raw-bit indexing, but the
   *encode* side (`intEncodeFn`) may clamp differently than hardware for
   out-of-range segments — the −1 suggests a saturated/negative encode path.

## Proposed correction plan (NOT applied — needs careful scoping)
Keep it minimal and unit-by-unit; do **not** restructure the PWL framework.

1. **Unit probes first** (temporary Scala sims, no library change):
   - `ExpOp(SInt(16))` on `{0, -1, -117, +5}` vs `pwl_int` golden.
   - `ReciprocalOp(SInt(16))` on `{1, 2, 4}` (expected `1, 0, 0` under current
     quantization conventions) and on a negative value.
   - `Softmax1D(SInt(16), channels=2)` fed `[0,-153],[-42,9]`: expect one-hot +1.
   This isolates whether the inversion is born in exp, reciprocal or the final
   multiply.
2. **Likely minimal fix**: compute the segment index on the **magnitude** (or
   clamp negative inputs to the lowest segment) inside the int branch of
   `ExpOp`/`ReciprocalOp`, mirroring what the math actually means
   (`exp(x≤0) ∈ [0,1]`, `1/sum(x>0)`). Keep the golden `pwl_int` synchronized
   in the same commit if bit-exactness must be preserved, otherwise update it
   to the corrected semantics.
3. **Validation gate**:
   - single-head `attention_i16` cosine must move from −1.000 to ≈+1;
   - `multihead_i16` full pass at seed 42 and one more seed;
   - re-run all other dtype suites (regression: the fix must be gated to the
     `SInt` branch only, float paths untouched);
   - Scala suite green (incl. any PWL unit tests).

## Workaround
None currently; `multihead_i16` is left failing / could be marked `xfail` with a
reference to this file until fixed.
