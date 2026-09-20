# Tiling & Resource Scaling — Design Anticipation

> **Status**: future-work design note (nothing here is implemented beyond the
> K axis). Captures the tile/slice vocabulary and the resource-scaling
> direction agreed during the S2/S3 compute-side spill closure (Sept 2026).
> **Audience**: anyone touching `LayerSpec` / `MatmulOp` / `Sequential` sizing,
> the folded-core refactor (Phase 3), or the board/planner plumbing.
> **Related docs**: `ddr_final_impl.md` (compute-side spill S0-S3),
> `ddr_impl.md` §5.3 (generic spill), `wave6_ddr_scaling_plan.md` P1/P3,
> `full_roadmap.md` §5, `roadmap_board.md` §4, `roadmap.md` pillar 5.

## 1. Why this exists

Today every layer instantiates its own datapath sized *by its shape*
(`Sequential` sizes weight DMA lanes as `Linear -> inFeatures`,
`Conv2D -> K²`, attention -> `embedDim`). A 4096-input Linear then costs 4096
parallel lanes. This is fine for MNIST-scale models and blocks everything
bigger: edge FPGAs and ASICs need operations **shaped to a resource budget**,
with weights streamed through a fixed-width datapath.

The tile/slice model is the vocabulary and the knob set that makes this
possible: it separates the **hardware shape** (LUT/DSP) from the **data
blocking** (BRAM/DDR traffic) so a model can be mapped onto a board budget
instead of dictating silicon area per layer.

This is a **prerequisite for the Phase-3 folded core** (`roadmap.md` pillar 1):
one physical compute engine time-multiplexed across layers needs (a) a fixed
shape, (b) per-layer descriptors, (c) a planner that proves the whole model
fits the board.

## 2. Vocabulary

### 2.1 Spatial vs temporal

- **Spatial**: duplicate hardware. More multipliers -> more **LUT/DSP**, fewer
  cycles. (Today: `lanes`, `parallelN`.)
- **Temporal**: reuse the same hardware across iterations. **LUT flat**,
  more cycles, smaller working set. (Today: N/M/K loops.)

### 2.2 Three granularity levels per axis

Every GEMM axis (M rows, K depth, N columns) can be described at three levels:

| Level | Meaning | Cost driver | Today (K axis) |
| :--- | :--- | :--- | :--- |
| **Chunk** | elements processed per cycle per MAC unit | LUT/DSP + AXI beat framing | `lanes` (= `weightLanes`) |
| **Tile** | hardware instantiated for one iteration | **LUT/DSP** | fixed (1 column x `lanes`) |
| **Slice** | data block loaded per pass/loop | **BRAM / DDR traffic** | `spillKSlice` |

Composition per axis: `axis = slice x sub-iterations` and `slice = q x tile`
(q >= 1). Buffers size to the **slices**, multipliers to the **tiles**.

### 2.3 Physical tile vs informational slice

Two distinct things, deliberately decoupled:

- **Physical tile** = a hardware shape. Changing it **costs/returns LUT/DSP**.
- **Informational slice** = a data block. Changing it **costs/returns
  BRAM pressure and DDR traffic** (it never changes LUT by itself).

They only coincide when a slice is realized spatially. In a fully temporal
engine the hardware tile is fixed, so slicing only moves memory/traffic. This
is why, today, `spillKSlice` changes BRAM and DDR traffic but **not** LUT.

### 2.4 Worked example (K axis, existing)

K = 64, `lanes` = 2, I8, 64-bit AXI beat:

| `spillKSlice` | Passes P = K/Ks | B buffer (Ks x N) | A re-streams | LUT |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 8 | smallest | 8x | unchanged |
| 32 | 2 | 4x bigger | 2x | unchanged |
| 64 | 1 (legacy) | biggest | 1x | unchanged |

Larger slice => fewer passes and less traffic, but more BRAM; LUT unaffected.

## 3. The three axes

| Axis | Semantics | Bounds (on-chip) | Bounds (DDR) | Output |
| :--- | :--- | :--- | :--- | :--- |
| **K** (depth) | accumulation across passes; the only axis needing partials + RAW fencing + bias-once | B buffer `Ks x Ns` | partial RMW `2*(P-1)*M*N` | native (rows) |
| **N** (columns) | independent column blocks; no accumulation | B buffer `K x Ns` + MAC tree width if spatial | A re-read `N/Ns` times | **strided** M x Ns per block -> assembly needed |
| **M** (rows) | independent row blocks | accumulators `(M window) x Ns`, spill region `Ms x Ns` | W re-read `M/Ms` times (unless resident) | native (contiguous rows) |

Key asymmetries:

- **K is special**: it is the only *reduction* axis. It owns the DDR partials,
  the inter-pass fence, the seed/drain pair and the final-pass-only bias.
  N/M blocking produce independent results and need none of that.
- **N slicing is the LLM axis** (huge `outFeatures`: vocab projections, FFN
  up-projections) where `K x N` weights cannot be buffered; its cost is the
  output assembly (M x Ns segments scattered into the M x N frame) and A
  re-reads.
- **M slicing is the sequence/batch axis**: output stays contiguous, A slicing
  is trivial, W is the re-read. It bounds the spill region and the output
  frame granularity, not the B buffer.
- **`temporal` is already a temporal M window**: it bounds the accumulator
  table to `min(temporal, M) x N` slots, not the spill region.

## 4. Decomposition model & constraints

The knobs are conceptually independent but coupled by five constraints:

1. **Divisibility**: `slice | axis` (no remainder), `tile | slice`
   (under-utilisation otherwise, `q >= 1`), `lanes | Ks` (replica fold-order
   contract), and `Ks * Ns * dtypeBytes % beatBytes == 0` (AXI framing).
2. **Buffers**: the weight buffer must hold `Ks x Ns`; the accumulator table
   holds `(M window) x Ns`. Slices, not tiles, size the memories.
3. **Bandwidth**: `K/Ks` (or `N/Ns`) multiplies A re-reads; `M/Ms` multiplies
   W re-reads. Small slices trade BRAM for DRAM traffic.
4. **Numeric order**: the per-output-element summation order is a
   **bit-exactness contract** (replica/`MatmulSpillTest`). K chunk order and
   pass order are fixed by `lanes` and `Ks`; M/N blocking does not change the
   per-element order (each `y[m,n]` stays independent).
5. **Framing**: slices start beat-aligned; a slice spanning a partial word is
   illegal in the current fetch plane.

Consequence: the planner treats `(mSlice, kSlice, nSlice, mTile, nTile,
lanes)` as a **small constraint system**, not a set of free integers.

## 5. Resource model (target: R0)

Per axis, the knob -> resource mapping the project must be able to compute and
report at elaboration time:

| Knob | Primary resource | Secondary cost |
| :--- | :--- | :--- |
| `lanes` / `kChunk` | LUT/DSP (MAC width, `DspMul` count) | AXI framing |
| `nTile` | LUT/DSP (`nTile` MAC columns) | bank conflicts, fanout |
| `mTile` | LUT/DSP (`mTile` MAC rows, Option 2 only) | A/B broadcast |
| `mSlice` | spill region `Ms x Ns` (DDR), accumulators | W re-read `M/Ms` |
| `kSlice` | B buffer `Ks x Ns` (BRAM) | A re-read `K/Ks`, partial RMW |
| `nSlice` | B buffer `Ks x Ns` (BRAM) | A re-read `N/Ns` |
| `temporal` | accumulator FF/BRAM `min(temporal,M) x Ns` | rows in flight |

Cycle/traffic estimates:

- compute cycles ~ `M * (K/lanes) * (N/(nTile))` (Option 1) or
  `(M/mTile) * (N/nTile) * (K/lanes)` (Option 2);
- DRAM traffic per inference ~ `K*N` (weights, once) + `re-reads * A` +
  `partial RMW` (K spill only).

The **flow link** is exactly this: **BRAM upper-bounds the slices** (working
set per pass), **DRAM bandwidth lower-bounds them** (refetch cost), and the
tile shape sets the compute rate. A slice is "good" only when both sides hold.

## 6. Current state — implemented today

| Feature | Status | Where |
| :--- | :--- | :--- |
| `lanes` / `weightLanes` (spatial K chunk) | **implemented** | `nn/LayerSpec.scala` (`effLanes`), `ops/matmul.scala` (`DspMul` arrays), per-beat framing |
| `spillKSlice` (temporal K slice, P passes) | **implemented + formally proven** | `LayerSpec.Linear.spillKSlice`, `SpillPassController`, `Sequential` slice fetch/window, `MatmulSpillTest`, S0-S3 notes in `ddr_final_impl.md` |
| DDR partial RMW (seed/drain, fence, bias-once) | **implemented** | `SpillPassController`, spill DMA pair, `Sequential` (`spillIn`/`spillOut`) |
| W slice DDR layout contract (slice-transposed) | **implemented (benches), tooling pending** | `SequentialSpillTest.programmedW`, §S2 closure note |
| `temporal` (temporal M window) | **implemented** | `ops/matmul.scala:310-326` (`min(temporal,M) x N` table), required for spill |
| `parallelN` (spatial N, mono-block) | **implemented in the engine, unused by `Sequential`** | `ops/matmul.scala:132`; `temporal`/`spill` require `parallelN=false` (`matmul.scala:58-61`) |
| `tileHeight` (activation band tiling for images) | **implemented** | `Sequential` image plane, `DMAReader2D` bands (a *different* tiling layer: activations, not GEMM axes) |
| Memory plumbing (MemorySpec, CSR 0x34, residency/prefetch, fit) | **implemented** | `nn/MemorySpec.scala`, `nn/CsrMap.scala`, `Accelerator.reportFit` |

**Not implemented**: N slicing, M slicing, configurable spatial tiles for
M/N, unified tiling API, the resource reporter, the board-driven planner.

## 7. Planned architecture

Two candidate directions (decision **open**):

- **Option 1 — weight-stationary extension** (natural continuation):
  spatial = `nTile x lanes`; M and K stay temporal. `nSlice` buys LUT/BRAM,
  `mSlice` buys memory only. Closest to today's proven engine.
- **Option 2 — output-stationary 2D** (full model):
  spatial = `mTile x nTile x lanes`; K temporal. Both `mSlice` and `nSlice`
  buy LUT/cycles. This is a **new engine** (A/B broadcast, per-tile
  accumulation) and re-opens the spill numeric-order contract.

The Phase-3 folded core (`full_roadmap.md` Phase 3, `roadmap.md` pillar 1)
pulls the architecture in one direction:

- **one core shape** sized for the largest layer (or per op-class),
- layers become **descriptors** (dims, addresses, slice counts),
- under-utilised smaller layers run on the same core.

This **inverts tile ownership**: today tiles are a per-layer property
(one engine per layer); in the folded core the tile is a **global core
shape**, and only slices remain per-layer descriptor fields.

## 8. High-level API design (target)

Three layers, no six loose parameters per layer:

```
model level   : TilingPolicy / ResourceBudget   // global budgets + defaults
layer level   : tiling: Option[Tiling] = None   // explicit overrides only
                Tiling(mTile, nTile, kChunk, mSlice, kSlice, nSlice) // -1 = auto
elaboration   : planner resolves -> resolvedTiling per layer + reportResources
```

- **Budget is global** (the board's physical BRAM/LUT/DSP/bandwidth is one
  shared pool); **values are resolved per layer** (they depend on the layer's
  geometry, dtype and alignment).
- Users write the policy once; per-layer overrides exist for tests and
  atypical layers; otherwise the planner fills the values.
- `spillKSlice` becomes the compatibility alias of `Tiling.kSlice`.
- In the folded-core world, the resolved values travel as **layer
  descriptors**; the tile part converges to the core shape.
- Elaboration must expose a `reportResources` (like `reportFit`) and fail
  fast on infeasible combinations.

## 9. Board integration & the automatic planner

Physical facts are **static per board** and belong in `boards/*.json`
(they already are, partially):

| Piece | Where today |
| :--- | :--- |
| LUT/FF/BSRAM/DSP counts | `boards/<slug>.json` `limits` (parsed by `board.py:154-158`) |
| On-chip words + address bases | `memory.onchip_words`, `memory.onchip_addr` |
| DRAM descriptor | `memory.ddr` (`present`, `size_bytes`, `dataWidth`, `max_burst`) |

Gaps to close for a planner:

1. **BRAM in bytes**: `limits.bram` is a *block count* (46 for GW2A-18); the
   planner needs usable bytes (block size x count, minus margin) ->
   add `bram_bytes` (or per-block kbits) to the board JSON.
2. **DRAM effective bandwidth + latency**: needed to lower-bound slices ->
   extend `memory.ddr` with e.g. `bytes_per_cycle` / `latency_cycles` once
   Phase-5b calibration lands (estimates until then).
3. **Safety margins** are *tooling policy*, not board facts: keep them as
   code defaults (optionally overridable), separate from `limits`.

Flow: `spinalml build --board X` loads the board facts, builds the
`TilingPolicy`, runs the planner (model + board -> per-layer resolved tiling),
elaborates, reports resources and fails fast when the model cannot fit — the
same spirit as `MemorySpec.reportFit` and `ddr_impl.md` Phase 3.

Board JSON carries **capacity, not performance promises**: Fmax/routing
depend on the design; synthesis numbers can later populate a separate
`measured` section.

## 10. Phases

| Phase | Deliverable | Gate |
| :--- | :--- | :--- |
| **R0** | Resource model + `reportResources` (pure Scala, no RTL): `(mTile,nTile,lanes,Ks,temporal,pipeline) -> LUT/FF/BRAM/DSP/cycles/traffic` | calibrated on 2-3 Tang syntheses, reported per layer |
| **R1** | Unified `Tiling` on `Linear`/`MatmulOp` + `TilingPolicy`, centralized `require`s, `spillKSlice` alias | legacy models elaborate byte-identically; invalid combos fail fast |
| **R2** | Axis implementations: N first, then M (Option 1) or the tiled engine (Option 2) | bit-exact e2e per axis (oracle + replica), scale + rerun, controller formal extended |
| **R3** | Board-driven planner: budget -> per-layer tiling, fail-fast, report | `build --board X` refuses infeasible models with actionable messages |

## 11. Open questions

1. **Option 1 vs Option 2** (weight-stationary extension vs 2D tiled engine):
   affects LUT ownership of M and the spill engine rewrite.
2. **Priority target**: vision (YOLO on Tang, Phase 2/3) vs LLM/transformer
   (N slicing, folding L2) — sets the first axis to implement.
3. **Planner heuristic**: bandwidth-first or BRAM-first allocation; how to
   split the global budget across layers (by MAC share, latency target, or
   uniform per op-class).
4. **Scope**: tiling axes on GEMM-family ops only (`Linear`, Conv via im2col,
   attention projections) or all ops.
5. **Board JSON additions**: `bram_bytes`, `ddr.bytes_per_cycle` /
   `latency_cycles` — schema + timing (Phase 5b).
6. **Residency interaction**: spill x residency/prefetch is forbidden today
   (`STREAM_PER_PASS`); the planner must encode that constraint.
7. **Numeric contract**: every new axis must preserve the per-element
   summation order proven by the replica (`Ks % effLanes == 0` generalization).

## 12. References

- `ddr_final_impl.md` — compute-side spill S0-S3 (K axis, implemented).
- `ddr_impl.md` §5.3 — generic spill, runtime contract, fencing.
- `wave6_ddr_scaling_plan.md` P1 (advanced tiling) and P3 (resource scaling).
- `full_roadmap.md` §5 (advanced memory architecture) and Phase 3.
- `roadmap.md` pillar 1 (folded core), pillar 4 (edge SLM).
- `roadmap_board.md` §4 (memory per board).
- `nn/LayerSpec.scala`, `ops/matmul.scala`, `nn/Sequential.scala` — current
  sizing sites.
