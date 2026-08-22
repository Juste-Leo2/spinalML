# Bug Detection Playbook — stream/pipeline debugging methodology

Step-by-step method used to localize the attention bugs of August 2026
(see `2026-08-attention-wxay-session.md`) and to diagnose the open
`int16-softmax-polarity.md` defect. It requires **no modification of the
library code**: everything runs from temporary scripts (`/tmp/...`) plus the
existing test infrastructure.

The core idea: golden-model instrumentation narrows *what* is wrong, then a
**runtime waveform (VCD)** pinpoints *which stage* replays/drops/corrupts beats,
and the generated Verilog explains *which register* does it.

---

## Step 0 — Reproduce deterministically

```bash
# pick ONE failing test, fixed seed, save logs to a file
SPINALML_SEED=42 .venv/bin/pytest tests/python/test_multiheadattention.py \
    -k "quant_w8a16" --debug-math -s 2>&1 | tee /tmp/opencode/logs/run.log
```

- `SPINALML_SEED` drives `random.seed` in the cocotb process (`utils/tb_utils.py`).
- A failure that depends on the seed is **data/timing dependent** — expect a
  handshake or saturation issue, not a fixed structural inversion.
- Always `tee` to a file; grep it later instead of re-running.

## Step 1 — Dump the golden chain per trial (`--debug-math`)

Add temporary `ATTDBG` lines in the failing cocotb test (they only write when
`DEBUG_MATH=1`, i.e. `--debug-math`):

```python
from utils.math_metrics import log_math_line
trial_idx = len(collect["out"]) if collect is not None else 0
log_math_line(f"ATTDBG t{trial_idx} X={X}")
log_math_line(f"ATTDBG t{trial_idx} Y_hw={Y_out}")
log_math_line(f"ATTDBG t{trial_idx} Y_gd={Y_expected}")
log_math_line(f"ATTDBG t{trial_idx} abs_err={errs}")
```

Output lands in `tests/true_math_errors.log` (gitignored). Per-trial dumps
reveal the failure pattern (which row, which trial, sign/scale of the error).

## Step 2 — Offline hypothesis hunt (numpy, no simulation)

Replicate the exact seeded draws and the golden chain, then compare the HW
output against candidate defects:

```python
import sys, random; sys.path.insert(0, 'tests/python')
from golden_models.dtypes import BF16
from golden_models.ops import dequant_hw, matmul_hw, softmax

random.seed(42)                      # same order of get_random_tensor calls!
X  = ...; Wq, Wk, Wv, Wo = ...       # replicate the test's draw sequence
# build candidates: norm, shifted rows, transposed V, stale-tile mixes...
# score each: max|candidate - Y_hw|
```

Two techniques that cracked the session:

- **Brute-force candidate space** (rows shifted by one, `Q·K` vs `Q·Kᵀ`,
  `V` vs `Vᵀ`, per-column stale mixes between tiles, prev-tile probs…).
  A diff of ~0.001 against one candidate = smoking gun; a winner that only
  fits on failing seeds and not on passing seeds confirms the mechanism.
- **Inverse problem**: when no candidate fits, reconstruct what the HW must
  have consumed: `P_used ≈ Y_hw · pinv(V·Wo)` then interpret the reconstructed
  probabilities (one-hot? shifted? wrong sign?).

## Step 3 — Generate the netlist (.v)

```bash
bash ./mill spinalML.test.testOnly spinalML.layers.MultiHeadAttentionTest -- -z w8a16
# -> MultiHeadQuantTestComp.v (+ MultiHeadQuantTestComp.v_toplevel_*.bin ROMs) at repo root
```

Note: pytest deletes root `*.v`/`*.bin` after each test (`cleanup_verilog`
fixture) — regenerate right before manual work.

Useful greps: module inventory (`grep -n "^module" Top.v`), instance wiring,
FSM constants. Read the Verilog *after* the VCD, to explain an already-localized
anomaly — not to fish blindly.

## Step 4 — Capture a runtime waveform (VCD) without touching the RTL

`run_layer_sim` already compiles Verilator with `--trace`, but never passes the
**runtime** flag. Re-run the exact failing testcase from a temporary script
(keep it in `/tmp`, never in the repo):

```python
# /tmp/opencode/trace_mha.py
import sys, os, glob, shutil
sys.path.insert(0, os.path.abspath("tests/python"))   # ABSOLUTE: sim CWD differs!
from cocotb_test.simulator import run

SIM_DIR = "/tmp/opencode/sim_mha_trace"
shutil.rmtree(SIM_DIR, ignore_errors=True); os.makedirs(SIM_DIR)
for f in glob.glob("MultiHeadQuantTestComp.v_toplevel_*.bin"):   # ROMs are mandatory
    shutil.copy(f, SIM_DIR + "/")

try:
    run(simulator="verilator",
        verilog_sources=[os.path.abspath("MultiHeadQuantTestComp.v")],
        toplevel="MultiHeadQuantTestComp",
        module="test_multiheadattention",
        testcase="cocotb_multihead_quant_w8a16",
        sim_build=SIM_DIR,
        timescale="1ns/1ps",
        extra_args=["--trace", "-Wno-fatal", "-Wno-WIDTH"],  # build flags
        plus_args=["--trace"],                               # RUNTIME flag -> dump.vcd
        extra_env={"DEBUG_MATH": "0", "RANDOM_SEED": "42"})
except SystemExit as e:
    print("SIM FAILED AS EXPECTED:", e)
```

Gotchas learned the hard way:
- Verilator runtime uses `plus_args`, **not** `sim_args`.
- `PYTHONPATH` is rebuilt from the parent's `sys.path` → insert **absolute**
  path, or the test module import fails inside the sim.
- Copy the `*.bin` ROMs into `sim_build` or `$readmem` warnings will silently
  zero your LUTs.
- Set `RANDOM_SEED` to keep cocotb's internal seeding identical to pytest.

## Step 5 — Parse the VCD: reconstruct beat streams per stage

Stdlib parser (no extra dependency). Two hard-won details:
- the first `$scope` has an **empty name** — guard `scope.append`, or every
  full name gets a garbage prefix;
- sample on **posedges** and count *fires* (`valid && ready`), do not try to
  reason from a single cycle snapshot: Verilator dumps post-settling values.

```python
id2sig, scope = {}, []
for line in open("dump.vcd"):                      # header pass
    ...                                             # $scope/$upscope/$var
# build CUTS = [(label, "softmax1D_2.io_x_stream"), ("softmax.y", ...), ...]
# for each cut: valid/ready ids + payload lane ids (sign/exponent/mantissa or raw bits)
# value-change pass: apply changes; on clk rising edge, if valid&&ready -> decode payload
```

Print the beat sequence at each cut of the datapath, e.g. for the attention:

```
softmax.x -> softmax.y -> ctx.C -> repack.C -> concat.C -> TOP y
```

**The first cut showing the anomaly localizes the guilty stage.**
Decisive patterns:
- `[P0, P1, P1, P2]` (bit-exact duplicate, one row lost) = register replay;
- clean input + corrupt output = bug inside that stage;
- corruption only from tile 2 = boundary/backpressure interaction.

## Step 6 — Drill into the guilty stage's handshakes

All internal signals are exposed in the VCD (`finalSyncValid`, `carryExpStream_*`,
`sumExpStream_*`, `expComp.io_*`, fork `io_outputs_*_fire`, `*_m2sPipe_*`…).
List them, then dump a posedge table over the failing window:

```python
SIGS = [("sumExp.v","sumExpStream_valid"), ("sumExp.r","sumExpStream_ready"),
        ("carryE.v","carryExpStream_valid"), ("invSum.v","recipComp_io_c_stream_valid"),
        ("y.v","io_y_stream_valid"), ("y.r","io_y_stream_ready"), ...]
# one row per posedge, dedupe identical consecutive rows
```

Reading example (softmax bug): `y.v && y.r` fired twice with the same payload
while the source pair had already advanced → the output pipe register was
re-emitted → then inspect the join's `ready` wiring in the Scala/Verilog.

## Step 7 — Confirm in the generated Verilog

Locate the module (`grep -n "^module MatmulOp (" Top.v`), then read the FSM
`case(_zz_N)` state transitions and counter widths. Things to check:
- every `Stream` join: is `ready` driven by the **pipe input ready**
  (`!full || sink.ready`) or by the raw sink ready? (Bug 2);
- counter widths vs their ranges (silent wrap = beat loss/duplication);
- matmul conventions: A-input row-major, B-input column-major, C row-major —
  any direct matmul→matmul B connection is an *implicit transpose*
  (Bug 1 / the earlier V bug).

## Rules of thumb

1. Never debug with `StreamFork` observation probes in Scala: they **change the
   timing** under test (learned the hard way). Prefer the VCD — zero intrusion.
2. Bit-exact duplicated beat = a payload register presented twice; hunt for a
   `valid`/`ready` contract violation, not for math.
3. Compare a **working** design's netlist with the failing one (same modules,
   different parameters) — but only after the VCD told you where to look.
4. Keep every debug artifact out of the repo: scripts in `/tmp/opencode/`,
   logs in `/tmp/opencode/logs/`, and remove `ATTDBG` instrumentation once the
   bug is closed.
5. Re-run the full Scala suite + the untouched dtype suites after any fix:
   handshake fixes can shift timing for everyone (they must, however, never
   change math).
