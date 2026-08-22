# I16 (SInt) Softmax Polarity Bug — _PATCHED_

## Summary
The integer-domain softmax path (`SInt` activations wider than 8 bits, i.e. the
PWL branch of `ReciprocalOp`) produced a **sign-inverted "hot" probability**
and saturating garbage on later tiles. This was a **pre-existing** defect,
masked for months because I16 attention outputs are tiny and the tests use a
`%FS` tolerance.

## Symptoms (before the fix)
| Test | Observation |
|---|---|
| `cocotb_multihead_i16` (seed 42) | FAIL at `Y[1][1]: got 32699.0 instead of 69.0`, deterministic |
| `cocotb_attention_i16` (single-head, seed 42) | PASSED tolerance but reported **Cosine = −1.000**, MAE 10.0 — systematically anti-correlated with golden |

Cosine −1 in single-head betrayed a systematic polarity inversion; multi-head
was the same defect amplified by larger accumulated values.

## Evidence (VCD beat capture, seed 42)
Golden chain for trial 0 head 0:

```
scores row0 = [0, -153]   -> golden probs = [1, 0]
scores row1 = [-42, 9]    -> golden probs = [0, 1]      # one-hot, +1
```

Hardware softmax beats: `(0, 0, -1, 0)` — hot lane **−1**; trial ≥ 1 showed
saturations (`32767`, `-32749`). `I16.from_float(1.0) == 1`: no encoding
ambiguity, a genuine sign bug.

## Root cause (final — different from the initial hypothesis)
The PWL framework fits each segment with `y = a·x + b` and stores **a and b as
integers of the same width as the data** (`MathLUTs.intEncodeFn` clamps to
±2^(w−1)). For `f(x) = 1/x` on segment `[0, 255]`:

```
y_start = 1/(0+ε) ≈ 1e9          y_end ≈ 1/255
a       = (y_end - y_start)/255 ≈ -3.92e6   -> clamps to -32768
b       = y_start                -> clamps to +32767
```

so the hardware evaluated

```
recip(x) = x·(-32768) + 32767     =>     recip(1) = -1      (!)
```

Softmax sums are one-hot in this domain (`exp(x≤0)` correctly yields one `1`
plus zeros), so every probability row came out as `-one-hot`. The initial
suspicions (raw-bit segment indexing of exp, negative-input handling) were
ruled out during unit analysis: `exp(int)` is correct on the softmax domain;
the reciprocal coefficients were the sole culprit.

## Fix (_PATCHED_)
1. `spinalML/src/spinalML/utils/PWL.scala`: new
   `PWLLUTs.createConstantSegmentFn(bitWidth, indexBits, mathFn)` — a
   **piecewise-constant** approximation storing `(0, f(sample))` per segment,
   where `sample` is the first abscissa of the segment with `|x| ≥ 1`
   (segment `[0,255]` samples at `x=1`, keeping `recip(1)=1` exact).
2. `spinalML/src/spinalML/ops/reciprocal.scala`: the SInt > 8-bit branch now
   uses it. The UInt branch and all float paths are untouched; I4/I8 keep the
   exact LUT path.
3. Golden synchronized:
   `tests/python/golden_models/ops.py::pwl_reciprocal_int` mirrors the new
   semantics bit-exactly.

Resource cost: zero delta — same ROMs, same pipeline; only ROM contents changed.

## Known limitation (accepted)
Within segment `[0,255]` the approximation is constant `1`: a softmax row whose
exponent sum ties at `s ∈ [2..255]` gets `recip = 1` where the idealized golden
uses `round(1/s) = 0` (two-hot vs all-zero probabilities). The induced output
error is bounded by one extra `V` row contribution (~%FS fractions), well inside
the `%FS` test tolerances; unique-max rows (the common case) are exact.

## Validation
- `test_reciprocal.py` I8/I16 ✅ (golden-synced), `test_softmax.py`,
  `test_div.py` ✅
- Attention suites seed 42: **22/22**, with
  - `ClassicalAttention I16`: Cosine **−1.000 → +1.000**, MAE 0
  - `MultiHeadAttention I16`: FAIL → pass, Cosine +1.000, MAE 0
  - I8 paths unchanged (LUT domain untouched)
- Spot check seed 1337 (mh w8a16/w4a4, classical w4a4/i16): 4/4 ✅
- Full Scala suite: 242/242 ✅
