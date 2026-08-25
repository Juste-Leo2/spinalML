# spinalML Symbolic Test Playbook

> **Purpose**: this document is the complete, self-contained reference for writing
> formal (symbolic) tests for any spinalML operation. Give it to any engineer — or
> AI agent — and it must be able to produce a correct, working `*Formal.scala` spec
> on the first try, verified against the exact SpinalHDL 1.14.2 API used here.
>
> Companion documents: [symbolicTest.md](symbolicTest.md) (installation + general
> concepts), [roadmap.md](roadmap.md) §8 (status map).

---

## 1. The pipeline (what actually happens)

```
YourScalaSpec (*Formal.scala)
   │  FormalConfig ... .doVerify(new X, "label")
   ▼
SpinalHDL generates SystemVerilog + a <Top>.sby config file
(no hand-written .sby / Verilog needed)
   ▼
formal/<Top>/rtl/<Top>.sv          ← generated RTL (gitignored)
formal/<Top>/<Top>.sby             ← generated config
   ▼
sby: base   → yosys read -formal; prep -top <Top>     (parse + symbolically elaborate)
     prep   → yosys prep flow (formalff, write_jny/write_rtlil, smt2)
     engine → yosys-smtbmc -s cvc4 (basecase + induction, or bmc steps)
   ▼
PASS (exit 0)  or  FAIL (exit 1 → counterexample VCD in formal/<Top>/<Top>_prove/engine_0/)
```

The solver (CVC4) receives the circuit as formulas and **decides** whether the
assertions hold for **every** legal input sequence — no sampling, no vectors.

---

## 2. Non-negotiable repo conventions

| Rule | Detail |
|---|---|
| **Location** | `spinalML/test/src/spinalML/symbolicTest/<category>/XxxFormal.scala`, `<category>` mirrors the source package (`ops/`, `dtypes/`, `layers/`, `poolings/`, `activations/` …) |
| **File name** | **must end with `Formal.scala`** — the CI discovers specs with glob `find spinalML/test/src/spinalML/symbolicTest -name "*Formal.scala"` |
| **Class name** | `class XxxFormal extends Component` (convention `op + "Formal"`) |
| **Package** | `spinalML.symbolicTest.<category>` — the CI derives the class name from the path (`spinalML.symbolicTest.ops.XxxFormal` ⇒ the file lives in `ops/`) |
| **Object** | `object XxxFormal { def main(...) }` with the `FormalConfig` bootstrap (required: it is what `runMain` invokes) |
| **DUT** | wrap the existing **test component** (e.g. `AddTestComp`) in `FormalDut(...)`. Test comps live in `spinalML/test/src/spinalML/ops/XxxTest.scala` (`case class XxxTestComp[T <: Data](dataType: HardType[T]) extends Component`). Never modify production RTL |
| **Golden model** | express the expected result *in the spec* (Scala), bit-exact, mirroring `tests/python/golden_models/` semantics |
| **Run locally** | `PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin" ./mill spinalML.test.runMain spinalML.symbolicTest.ops.XxxFormal` — exit `0` = PASS, exit `1` = FAIL/counterexample |
| **Workspace** | `formal/` (gitignored). Kept only with `.withDebug` (cleaned otherwise) |
| **CI** | `.github/workflows/ci-symbolic.yml` runs all specs automatically: system deps → yosys ≥ 0.33 (else source build) → sby v0.68 → proof loop with `taskset -c 0,1,2,3` on the radxa |

---

## 3. API reference (verified against SpinalHDL **1.14.2** sources)

### 3.1 The DUT and symbolic drivers

| API | Meaning |
|---|---|
| `FormalDut(new X(...))` | Wraps the component under proof; marks the enclosing component as formal tester (auto-pull + async→sync reset remapping applied automatically) |
| `anyseq(sig)` | sig is a **free variable that can change every cycle** (use for streaming/valid/ready inputs) |
| `anyconst(sig)` | sig is a **free constant fixed for the whole run** (use for configuration inputs) |
| `allseq(sig)` / `allconst(sig)` | like anyseq/anyconst but explore *all* inputs in *parallel* (rare; equivalent for our widths) |
| `assumeInitial(b)` | constraint that must hold **at reset** (bootstraps the state space) |
| `assume(b)` | clocked constraint on the **legal environment** (inputs/handshakes the op is permitted to see) |
| `assert(b)`, `assert(b, "message")` | the **property to prove** (clocked; under a `when` guard it must hold only when the guard is active) |
| `cover(b)` | reachability check: prove the solver *can* reach `b` (deadlock/never-valid detection) |
| `past(x, delay=1)` | value of `x` `delay` cycles ago (needs no special import; implemented via `RegNext` chain) |
| `pastValid()` / `pastValidAfterReset()` | helper flags: a previous cycle existed (after reset) |
| `rose(b)` / `fell(b)` / `changed(x)` / `stable(x)` | edge / change detectors on the previous cycle |
| `initstate()` | Bool that is **True only in the first cycle after reset** (sequential invariants) |

> **Rule**: every DUT input that is not internally connected **must** be driven with
> `anyseq`/`anyconst`/`assume`, otherwise SpinalHDL aborts with
> `NO DRIVER ON (...)`. Inputs constrained by `assume(...)` still need `anyseq` first.

### 3.2 The bootstrap (`FormalConfig`, from `spinal.core.formal._`)

```scala
FormalConfig
  .withSymbiYosys                                    // default backend; keep explicit
  .withProve(10)                                     // mode + depth: full proof (k-induction)
  // .withBMC(100)                                   // bounded model check: only steps ≤ depth
  // .withCover(50)                                  // reachability
  .withTimeout(600)                                  // seconds; inconclusive ≠ failed
  .withDebug                                         // keep formal/ workspace (traces)
  .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4))) // smtbmc engine (see below)
  .workspacePath("formal")                           // output dir (gitignored)
  .doVerify(new XxxFormal, "free_label")             // DUT by-name + label for logs
```

Verified facts (1.14.2):

- **Defaults if you omit**: no mode → `bmc` depth 100; no engine → `SmtBmc()` whose
  **default solver is Yices** — always set `SmtBmc(solver = SmtBmcSolver.Z3)` (z3 is the
  only solver installed in the project environment).
- Modes: `bmc` (bounded, steps ≤ depth), `prove` (k-induction: basecase at depth +
  induction at depth), `cover` (reachability). `live`/`equiv`/`synth` exist in the API —
  not used here.
- `SmtBmc(nomem, syn, stbv, stdt, nopresat, unroll, dumpsmt2, progress=true, solver)`
  maps to `yosys-smtbmc` flags; `progress` yields the `--progress cvc4` used in logs.
  Other engines: `Aiger()`, and `Abc()` — **do not use `Abc()`**: the project's yosys
  is built with `ENABLE_ABC := 0` (formal proof never needs ABC).
- `withTimeout(t)` sets the sby `timeout` (seconds) per task. A timeout gives
  *inconclusive* status — it does not fail the job.
- The DUT's **async resets are automatically rewritten to synchronous** resets
  (FormalPhase) — async reset never blocks a proof.
- `doVerify` generates SystemVerilog (`config.generateSystemVerilog`) + `<Top>.sby`
  in `formal/<Top>/`, then launches `sby -f`.
- Generated `.sby` script section is always:
  `read -formal <files>` / `prep -top <Top>` (+ `-ifx` if `withOutWireReduce`) —
  no `synth`, no `abc`: this is why the CI yosys build needs no ABC.
- Workspace layout: `formal/<Top>/<Top>.sby`, `rtl/<Top>.sv`, `<Top>_<mode>/` task
  dir with `engine_0/` (own `logfile*.txt`, `trace.vcd`, `trace_tb.v` on failure),
  and `model/design*.log` (yosys) — **when sby reports an opaque failure, read
  `model/design.log` / `model/design_prep.log` / `model/design_smt2.log`**; they
  contain the real error (`No such command: X`, syntax errors, …).

---

## 4. Anatomy of the reference spec (`AddFormal.scala`, ops/Add)

```scala
package spinalML.symbolicTest.ops

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.ops.AddTestComp
import spinalML.dtypes.I8

class AddFormal extends Component {
  // 1. The DUT under proof (test component, black box)
  val dut = FormalDut(new AddTestComp(I8()))

  // 2. Drive EVERY input with free symbolic variables
  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.payload)

  // 3. Start from a proper reset (never explore impossible initial states)
  assumeInitial(clockDomain.isResetActive)

  // 4. Legal domain (assume) — the environment the op is allowed to see
  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)

  // 5. Keep inputs stable across cycles so the pipelined (m2sPipe, +1 cycle)
  //    output can be compared against the same input pair
  assume(dut.io.a.stream.payload === past(dut.io.a.stream.payload))
  assume(dut.io.b.stream.payload === past(dut.io.b.stream.payload))

  // 6. SPEC (assert): the golden model, bit-exact, per lane
  val expected0 = (dut.io.a.stream.payload(0) + dut.io.b.stream.payload(0)).resize(8 bits)
  val expected1 = (dut.io.a.stream.payload(1) + dut.io.b.stream.payload(1)).resize(8 bits)

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {   // completed handshake
    assert(dut.io.c.stream.payload(0) === expected0, "Add lane 0 mismatch (I8)")
    assert(dut.io.c.stream.payload(1) === expected1, "Add lane 1 mismatch (I8)")
  }
}

object AddFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(10)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new AddFormal, "add_i8")
  }
}
```

Why each piece is there:

1. `FormalDut(...)` — inputs become symbolic; the component is treated as black box.
2. `anyseq(...)` on everything (see §3.1 — NO DRIVER ON aborts otherwise).
3. `assumeInitial(clockDomain.isResetActive)` — without it the solver explores
   garbage initial states and finds bogus counterexamples.
4. `assume(...)` — restrict to legal transactions. Without the valid/ready
   assumptions, the solver may drive an invalid handshake and "find" a bug.
5. Stability + `past(...)` — the op contains an `m2sPipe()` (+1 cycle of latency);
   the golden expression is evaluated on *current* inputs, so inputs must be frozen
   while the pipelined result compares against them.
6. Golden model under a `when` handshake guard — assert only on completed output
   transactions. Widths must match hardware truncation (`resize(8 bits)` = I8
   wrap-around, matching the RTL `valA + valB` on `SInt`).

---

## 5. The recipe: write any new spec in 10 steps

1. **Pick the target op** (`spinalML/src/spinalML/ops/X.scala`) and its test
   component (`spinalML/test/src/spinalML/ops/XTest.scala` → `XTestComp`). If a test
   comp does not exist, create `XTestComp[T <: Data](dataType: HardType[T])` that
   instantiates the op and exposes `slave()`/`master()` IO with the same
   `Tensor(dataType, shape, lanes)` interface.
2. **Read the op's RTL** to extract: input/output tensor shapes and lanes; stream
   handshakes; latency (any `.m2sPipe()`/`StreamJoin`/registers); data-path type
   dispatch (SInt/UInt/FloatML branches); truncation behavior.
3. **Determine the golden model** in Scala, **bit-exact** with the hardware:
   - SInt/UInt: plain `+`, `-`, `*`, `/`, … with explicit `.resize` to the dtype
     width (signed ops wrap; match the RTL exactly);
   - FloatML (FP8/FP16/BF16): use the same rounding path as the RTL
     (`spinalML.utils.Float.add(...)` etc.) — never native Scala float math;
   - elementwise over lanes: index `payload(i)` per lane.
4. **Instantiate**: `val dut = FormalDut(new XTestComp(TYPE()))` (dtype from
   `spinalML.dtypes.*`: `I8()`, `I16()`, `U8()`, `U4()`, `FP8()`, `FP4()`, `BF16()`,
   `FloatML(...)`).
5. **Drive everything**: `anyseq(...)` on all stream `.valid`, `.ready` and
   `.payload` fields.
6. **Reset**: `assumeInitial(clockDomain.isResetActive)`.
7. **Legal domain**: `assume(...)` valid/ready contracts; freeze inputs with
   `past(...)` equality when the output is pipelined; constrain configuration
   inputs with `anyconst` + assumptions if needed.
8. **Spec**: golden values; `when(output handshake) { assert(...) }` with a message.
9. **Bootstrap** (§3.2): `withProve(10)` for full proofs (sweet spot), `withBMC(100)`
   for bounded checks on big designs, `withTimeout(600)`, `withDebug`,
   `SmtBmc(solver = cvc4)`, `workspacePath("formal")`.
10. **Validate** (non-optional QA): run once with a deliberately broken assertion
    (`=== expected + 1`) → the solver MUST return a counterexample (exit 1, VCD in
    `engine_0/`). Restore the correct assertion, rerun → exit 0. This proves the
    spec actually detects bugs and isn't vacuously satisfied.

---

## 6. Mode selection (decision tree)

| Situation | Mode | Why |
|---|---|---|
| Combinational op, everything bounded, small dtype (I8/U8/I16…) | `withProve(10)` | full proof in seconds (k-induction), no depth limit semantics |
| Design too big / prove explodes/timeouts | `withBMC(100)` then grow depth | bounded but still exhaustive within depth |
| Prove *inconclusive* at higher depth | try `withProve(depth+10)`, `withTimeout` up, or split the spec | induction may need deeper basecase |
| Reachability: "can valid ever fire?", liveness, deadlocks | `withCover(50)` + `cover(cond)` | solver search for a state satisfying `cond` |
| Sequential/stateful block (counters, FIFO, DMA) | prove with invariants: `initstate()`, `past(...)`, `assumeInitial` on registers, `stable(...)`; fall back to `withBMC` | inductive proofs need a *closed* invariant set |

---

## 7. Beyond ops: flows, state and dtype units (decomposition doctrine)

The framework is **agnostic to the block type** — `FormalDut` + `anyseq`/`assume`/
`assert`/`cover` can target any component. What changes is the *kind of property*:

| Target | Property strategy (≠ op equivalence) |
|---|---|
| `ops/` (combinational, streamed) | Golden-model equivalence per lane (§4/§5) |
| `StreamDoubleBuffer` / stream fabric | **Flow invariants**: no data loss (every accepted `valid && ready` transaction eventually leaves), no invented data (output payload === accepted input payload), order preserved — via `past(...)` and guarded handshake assertions |
| DMA / `DMAReader(2D)` | Adversarial flow: `anyseq` on every readiness/validity, **bounded reachability** (`withBMC`) and **liveness/deadlock checks** (`withCover` + `cover(...)` that output `valid`/transfers can fire); external memory modelled with `anyconst` + assumptions |
| Any sequential/stateful block | **Closed invariants**: `initstate()`, `stable(...)`, `past(...)`, bounded counters — prove state stays in its legal set and outputs match it (§8.2) |

**Dtypes are types, not components** — so "verifying a dtype" means proving the
arithmetic/quantization **units that implement it** (`spinalML.utils.Float.add/...`,
`requantize` shift+round, `cast` wrap-vs-clip, ...). Rules:

1. **Prove units per concrete dtype instance** (`FP8_E4M3()`, `BF16()`, `I8()`, ...),
   **never generically** — the solver needs fixed widths (symbolic mantissa/exp bit
   counts blow up instantly).
2. **Units first, then ops, then layers** — the decomposition doctrine:
   - a unit proven once is a **certified lemma**; every op that instantiates it
     inherits it at proof time;
   - if an op proof fails while its unit is already proven, the bug is **by
     elimination** in the glue (lanes, indexing, truncation, handshakes) — no need
     to re-debug rounding/saturation semantics op after op;
   - quantization units (`requantize`, float add, expansion/clip) are the classic
     source of subtle, "real-looking" counterexamples — isolating them pays off
     across the whole `ops/` family.

---

## 8. Pitfalls and operational notes (hard-won)

- **`SymbiYosys failure` is an envelope, not a cause**: the real error is in
  `formal/<Top>/<Top>_<mode>/logfile*.txt` or `model/design*.log`. Always inspect
  before touching the spec.
- **Version matrix (project standard)**: Yosys **0.33** (git `2584903a060`), CVC4
  **1.8**, SBY **v0.68**. Yosys < 0.24 breaks sby's prep (`formalff -hierarchy`);
  the CI builds 0.33 from source if the system one is older (built once in
  `$HOME/.local` with `ENABLE_ABC := 0` — ABC is unused by `prep`+`smtbmc`, and
  building it takes ~20 min and overheats the runner).
- **PATH trap**: the `sby` launcher is `#!/usr/bin/env python3` and needs `click`/
  `yaml` in that Python. With a venv active, use the ordered PATH from §2
  (system Python first, then `$HOME/.local/bin`).
- **CI runner (radxa)**: proofs run on the 4 LITTLE cores
  (`taskset -c 0,1,2,3`); `MAKEFLAGS=-j1`; builds go in `$HOME` (never `/tmp`,
  which can be read-only). After an interrupted build on the radxa, **always**
  `make clean` — a hard power-loss mid-build silently corrupts object files and
  yosys then answers `No such command: <pass>`.
- **Do not use `Abc()` engine** — yosys is built without ABC; the engine would fail.
- **SpinalHDL bit-slicing trap**: `x(1, 2 bits)` extracts bits **[2:1]** (syntax is
  `(offset, width)`, not `(low, high)`) — an alignment check written this way produces
  real-looking counterexamples that are pure spec bugs. Cross-check every suspicious
  assertion against the generated SV (`formal/<Top>/rtl/<Top>.sv`) before suspecting
  the RTL: the assert operands appear there as explicit nets.
- **Comparison width truncation**: in a relational op between mismatched widths,
  SpinalHDL may satisfy it by TRUNCATING the wider side (`.resized` chains included)
  instead of growing the narrower one — a bound like `elemCnt < rowWords * EW` silently
  became `< (rowWords*EW)[1:0]`. Widen explicitly (`elemCnt.resize(12 bits) < wideExpr`).
- **Pull registers, not combinational nets**: gate assertions on `dut.fsm.stateReg`
  (as bits: `.pull().asBits.asUInt`, encodings = declaration order) and latched regs;
  reconstruct comb conditions in the spec. And remember per-row/per-phase bookkeeping:
  trackers like "first kept beat" must reset at EVERY phase boundary, not only per command.
- **State-space diet**: 32-bit address arithmetic dominates BMC solve time — shrink the
  DUT's `addressWidth` (16 is plenty for fetch-ordering proofs) and keep depths small
  (BMC ~10-15); a full serialized-row DMA proof then runs in seconds.
- **k-induction**: `prove` = basecase + induction. Both must pass.
- **Under-specified specs pass trivially**: missing `anyseq`/`assumeInitial`/
  domain assumptions are the usual cause of either aborts or bogus counterexamples.
- **Data width**: always `.resize` golden expressions to the dtype width; a width
  mismatch produces a real-looking counterexample.
- **Latency**: any registered stage (`m2sPipe`, `StreamJoin`+regs) shifts results —
  guard assertions with the output handshake or `past(...)`.
- **Timeout ≠ failure**: sby timeout → inconclusive; CI does not treat it as an error
  (it does not exit 1).

---

## 9. Copy-paste templates

### 8.1 Combinational streamed op, 2 inputs + 1 output, `lanes` lanes (default)

```scala
package spinalML.symbolicTest.<category>

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinalML.ops.<X>TestComp
import spinalML.dtypes.<T>

class <X>Formal extends Component {
  val dut = FormalDut(new <X>TestComp(<T>()))

  anyseq(dut.io.a.stream.valid)
  anyseq(dut.io.b.stream.valid)
  anyseq(dut.io.c.stream.ready)
  anyseq(dut.io.a.stream.payload)
  anyseq(dut.io.b.stream.payload)

  assumeInitial(clockDomain.isResetActive)

  assume(dut.io.a.stream.valid)
  assume(dut.io.b.stream.valid)
  assume(dut.io.c.stream.ready)
  assume(dut.io.a.stream.payload === past(dut.io.a.stream.payload))
  assume(dut.io.b.stream.payload === past(dut.io.b.stream.payload))

  val lanes = dut.io.a.stream.payload.length
  val expected = Vec(<T>(), lanes)
  for (lane <- 0 until lanes) {
    // Golden model, bit-exact with the RTL (match dtype semantics / truncation)
    expected(lane) := (dut.io.a.stream.payload(lane) + dut.io.b.stream.payload(lane)).resize(8 bits)
  }

  when(dut.io.c.stream.valid && dut.io.c.stream.ready) {
    for (lane <- 0 until lanes) {
      assert(dut.io.c.stream.payload(lane) === expected(lane), s"<X> lane $lane mismatch")
    }
  }
}

object <X>Formal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withProve(10)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new <X>Formal, "<x>_<t>")
  }
}
```

### 8.2 Sequential / stateful block (invariant style)

```scala
class <X>Formal extends Component {
  val dut = FormalDut(new <X>TestComp(...))
  anyseq(dut.io.in.payload)            // + anyseq on valid/ready
  anyseq(dut.io.out.ready)
  assumeInitial(clockDomain.isResetActive)
  assume(dut.io.in.valid)
  assume(dut.io.out.ready)

  // invariant: after the first cycle, the internal state obeys the spec
  val first = initstate()
  when(!first) {
    // e.g. state must stay within bounds, never overflow, etc.
    assert(dut.io.internal.cnt <= maxValue, "counter overflow")
  }

  when(dut.io.out.valid && dut.io.out.ready) {
    assert(/* output === function of past inputs */, "...")
  }
}
// same object/main as 8.1 (adjust withProve depth / timeout as needed)
```

---