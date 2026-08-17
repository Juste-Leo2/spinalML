# Symbolic Testing (Formal Verification)

spinalML uses **formal verification** in addition to the Python/Cocotb co-simulation
(`pythonTest.md`). While the co-simulation checks the hardware against the golden models
on a *sampled* set of vectors, symbolic testing **proves** — mathematically, for **all**
possible inputs and states — that the hardware satisfies the specification.

The stack used is:

- **[SpinalHDL Formal](https://spinalhdl.github.io/SpinalDoc-RTD/master/SpinalHDL/Formal)** — the specification is written in Scala, next to the tested component.
- **[Yosys](https://yosyshq.net/yosys/)** — parses the generated Verilog and builds a symbolic model of the circuit (signals become symbolic variables).
- **[SymbiYosys](https://github.com/YosysHQ/symbiyosys)** — orchestrates the solvers (here, Z3) that decide whether a property holds for all inputs, or produce a counterexample.

SpinalHDL generates the Verilog and the `.sby` configuration automatically: no hand-written
formal file is needed.

## Prerequisites and Installation (WSL / Ubuntu)

Tested procedure on WSL (Ubuntu 24.04): Yosys 0.33, Z3 4.8.12, SBY v0.68.

```bash
# 1. Yosys + Z3 SAT solver (system packages)
sudo apt-get update
sudo apt-get install -y yosys z3 python3-venv

# 2. SymbiYosys from the official repository (no setup.py anymore: make install)
git clone https://github.com/YosysHQ/symbiyosys.git /home/leo/symbiyosys
cd /home/leo/symbiyosys
make install PREFIX=$HOME/.local

# 3. sby runtime dependencies (Python launcher): click + yaml
sudo apt-get install -y python3-click python3-yaml

# 4. Put sby on the PATH and verify
export PATH="$HOME/.local/bin:$PATH"
yosys --version   # Yosys 0.33+
z3 --version      # Z3 4.8.12+
sby --version     # SBY v0.68+
```

> [!NOTE]
> Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` so `sby` is
> available in later terminals. Z3 is used as the SMT engine because Boolector
> is not packaged in the Ubuntu repositories.

> [!WARNING]
> The `sby` launcher uses `#!/usr/bin/env python3` and needs `click`/`yaml`
> in **that** Python. If a virtualenv is activated (e.g. the project's `.venv`)
> its Python shadows the system one and `sby` fails with
> `ModuleNotFoundError: No module named 'click'`.
> Solutions: run from a clean shell, or order the PATH so the system Python
> comes first, e.g.:
> ```bash
> PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin" ./mill ...
> ```
> (or install `click`/`pyyaml` into the venv).

> [!WARNING]
> **SBY v0.68 needs Yosys >= 0.24.** Its prep flow runs
> `formalff -setundef -clk2ff -ff2anyinit -hierarchy`, and the `-hierarchy`
> option only exists from Yosys 0.24 onwards. Debian bookworm (and the radxa
> self-hosted runner) ship Yosys 0.23, with which sby aborts with an opaque
> `SymbiYosys failure` (see `prep: ERROR: Command syntax error: Unknown option`
> in the sby logfile). The CI workflow requires the tested combo (>= 0.33):
> if the installed version is older it builds `yosys-0.33` from source once
> into `$HOME/.local` (persistent on the self-hosted runner). Keep local/CI
> versions aligned with Yosys 0.33 / Z3 4.8.12 / SBY v0.68.

## Testing Architecture

Formal specifications live in `spinalML/test/src/spinalML/symbolicTest/`,
mirroring the source packages in subfolders (`symbolicTest/ops/`,
`symbolicTest/dtypes/`, ...): one `XxxFormal.scala` per component, right where
the `tests/python/` golden models are described — but in Scala. The production
RTL is **never** modified: the specification reuses the existing test components
(`AddTestComp`, ...) as a black box.

### Anatomy of a formal specification (`AddFormal.scala`)

```scala
class AddFormal extends Component {
  val dut = FormalDut(new AddTestComp(I8()))      // 1. the DUT under proof

  // 2. Drive every DUT input with free symbolic variables (explore ALL values)
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.payload)

  // 3. Start from a proper reset state
  assumeInitial(clockDomain.isResetActive)

  // 4. Domain constraints (assume): legal environment of the op
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  // Keep inputs stable so the pipelined output can be compared
  assume(dut.io.a.stream.payload === past(dut.io.a.stream.payload))
  assume(dut.io.b.stream.payload === past(dut.io.b.stream.payload))

  // 5. Specification (assert): the golden model, bit-exact
  val expected0 = (dut.io.a.stream.payload(0) + dut.io.b.stream.payload(0)).resize(8 bits)
  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    assert(dut.io.c.stream.payload(0) === expected0, "Add lane 0 mismatch (I8)")
  }
}
```

Five things matter:

1. **`FormalDut(...)`** — wraps the component; its inputs become *symbolic variables*
   (every possible value is explored, nothing is sampled).
2. **`anyseq(...)`** — required: every DUT input must be driven, otherwise
   SpinalHDL aborts with `NO DRIVER ON (...)`.
3. **`assumeInitial(clockDomain.isResetActive)`** — the state space must start
   from a proper reset, or the solver explores impossible initial states.
4. **`assume(...)`** — restricts the solver to the *legal* domain. Without these,
   the solver finds invalid counterexamples (e.g. a transaction that starts while
   the handshake is not established).
5. **`assert(...)`** — the property to prove. Placed under a `when`,
   it is only required to hold when the guard is active (here: a completed
   handshake on the `m2sPipe` output, which adds one cycle of latency).

A good property is written from the **golden model's point of view**
(`tests/python/golden_models/`): express what the result *must be* as a function
of the inputs, and let the solver prove the hardware matches.

## Running a Formal Test

```bash
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin" \
  ./mill spinalML.test.runMain spinalML.symbolicTest.ops.AddFormal
```

The bootstrap (in the `main`) generates the Verilog and the `.sby` file,
then launches SymbiYosys. This is the exact bootstrap used (SpinalHDL 1.14.2):

```scala
import spinal.core.formal._

FormalConfig
  .withSymbiYosys                              // no parentheses: parameterless def
  .withProve(10)                               // unbounded proof, induction depth 10
  .withTimeout(600)                            // guard rail against solver explosion
  .withDebug                                   // keeps the formal/ workspace for inspection
  .withEngies(List(SmtBmc(solver = SmtBmcSolver.Z3)))  // SMT engine selection
  .workspacePath("formal")                     // outputs land in formal/ (gitignored)
  .doVerify(new AddFormal, "add_i8")           // by-name argument
```

API notes (1.14.2): use `FormalConfig` (not `SpinalFormalConfig`), `withSymbiYosys`
without `()`, and pass the spec by name to `doVerify`. The `formal/` workspace
(Verilog, `.sby`, solver logs, VCD traces) is gitignored; with `withDebug` it is
kept after a run, otherwise it is cleaned up automatically.

## Interpreting the Results

A successful proof finishes with an exit code `0` and, inside the sby log
(`formal/AddFormal/AddFormal_prove/logfile.txt`):

```
summary: engine_0 (smtbmc --progress z3) returned pass for basecase
summary: engine_0 (smtbmc --progress z3) returned pass for induction
summary: successful proof by k-induction.
SBY ... DONE (PASS, rc=0)
```

Meaning: **no** input combination (of the legal domain) can ever violate the
assertions — the equivalent of running the co-simulation over the *entire*
input space, in seconds (the `AddFormal` I8 proof completes in under a second).

If a property is wrong or the hardware has a bug, the run exits `1` and Yosys
reports a **counterexample** (VCD trace, witness, testbench) in
`formal/AddFormal/AddFormal_prove/engine_0/`:

```
##   Assert failed in AddFormal: AddFormal.sv:72.35-73.71 ($assert$AddFormal.sv:72$51)
##   Writing trace to VCD file: engine_0/trace.vcd
##   Writing trace to Verilog testbench: engine_0/trace_tb.v
##   Status: failed
```

The VCD shows the exact illegal input sequence — the same waveform format used
for the Python co-simulation.

> [!TIP]
> To validate that the workflow *detects* bugs (and not only proves trivia),
> the `AddFormal` spike is exercised with a deliberately broken assertion
> (e.g. `=== expected0 + 1`): the solver must produce a counterexample.
> Restore the correct property afterwards.

## Tips and Pitfalls

- **Engine / solver**: Z3 is the pragmatic default here. Boolector (faster on
  some proofs) can be added later as a second engine for CI.
- **Timeouts**: a proof that exceeds the timeout is reported as *inconclusive*,
  not failed — it does not block the pipeline.
- **Pipelined outputs**: `m2sPipe` (and any registered stage) adds latency:
  guard the assertions with the output handshake (`valid && ready`) or use
  `past(...)`.
- **Data widths**: match the truncation of the hardware exactly in the golden
  expression (e.g. I8 addition wraps via `.resize(8 bits)`).
- **State explosion**: prove *small blocks*, never whole systems at once.
  Combinational ops on 8/16-bit dtypes are the sweet spot; for sequential
  blocks use invariants (`past`, `initstate`) and bounded modes (`withBMC`)
  when `withProve` becomes too expensive.

## Roadmap

See section [7. Formal Verification](roadmap.md) of the roadmap: the `Add`
spike is step 1; the remaining primitive `ops/`, then stream/memory
invariants (`StreamDoubleBuffer`, `DMAReader`), and finally CI integration.