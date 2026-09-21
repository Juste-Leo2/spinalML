# Compute-side spill — Plan final d'implémentation (étapes S0-S3)

> **Statut** : plan figé le 19/09/2026. **S0, S1, S2 et S3 closes** (notes de
> clôture dans chaque section — S3 soldée par la PR « réplica spill »,
> commits R1-R6 sur `ddrImpl2` : layout slice-transposé, fold réplica,
> e2e `ModelReplica`, CLI minimale, trafic mesuré).
> **Docs liés** : `docs/ddr_impl.md` §5.3 (soudure compute), `docs/ddr_replica_status.md`
> (contrat numérique, non-régression), `docs/wave6_ddr_scaling_plan.md` P1 (contexte cap DRAM),
> `docs/roadmap_board.md` §4 (mémoire par board).
> **Décisions actées en revue** : K-split (le mur = buffer B `paddedK*N` en SRAM, pas la
> table `M*N`) ; host-transparent (single-START préservé, harness/CLI/UartSoC intacts) ;
> spill ⇒ `temporal >= 1` (drain `stateEmitRow` réutilisé) ; sommes pleine largeur + bias
> final unique + `K_slice % effLanes == 0` ⇒ **zéro changement réplica en v1**.
> **Extensions actées** : rejeueur A on-chip (A petit profond spillable, budget BSRAM) ;
> bias non-final = stream de zéros + gate test `+0` (fallback : bypass `BiasAddOp`).

## Rappel architecture (ce qu'on construit)

Par passe `p` sur `P = K / K_slice` tranches :
`W-slice (K_slice×N)` re-fetchée + `A` re-streamée (DDR-backed) ou rejouée (replay buffer)
→ `MatmulOp` ensemencé depuis `spillIn` (p > 0), drainé vers `spillOut` (p < P-1),
bias réel passe finale seule, zéros sinon → partielles `M×N` pleine largeur en DDR
entre passes via paire `DMAReader`/`DMAWriter` + curseur spill (reset sur write `0x34`).

## S0 — Knobs d'élaboration + `require`, zéro changement RTL (0.5 j)

- `LayerSpec.Linear` : `spillKSlice: Int = -1` (même pattern que `weightLanes`, défaut
  = legacy). `require` locaux : diviseur de `inFeatures`, multiple de `effLanes`.
- `Sequential` : nouveau param `spillReplayBudgetBytes` (défaut ~BSRAM-réaliste) ;
  par couche spillée, `require` : `temporal >= 1`, `!weightResidency` (STREAM_PER_PASS
  seul en v1), éligibilité A (nœud 0 DDR-backed **ou** `M*K*dtypeBytes ≤ budget`,
  sinon message pointant vers A-spill v2) ; expose `totalSpillBytes`
  (Σ `M*N*accWidth` beat-aligné).
- `Accelerator` : `reportFit` avec la valeur calculée (`model.totalSpillBytes`) ;
  `require` spill configuré ⇒ `memory.spillBase.isDefined`.
- **Gate** : `SpillConfigTest` (accepté/refusé aux bons endroits, fit qui lève) ;
  tout le reste vert, aucun comportement RTL ne change.
- **✅ VALIDÉE le 19/09/2026.** Écarts au plan actés pendant l'exécution :
  pas de `require(!weightResidency)` — le flag n'instancie que le plan de
  contrôle (inactif par défaut, et `Accelerator` l'active par défaut) ;
  l'interaction spill × residency est un **contrat runtime** (`STREAM_PER_PASS`,
  `CSR 0x10 = 0`) que le contrôleur S2 appliquera, pas un `require`
  d'élaboration.

## S1 — `MatmulOp` slice engine (1 j)

- Ports : `spillIn` (seed, p > 0), `spillOut` (drain, p < P-1, muxé sur `stateEmitRow`),
  `passFirst`/`passLast` (mux zéro-vs-spill à l'init, `spillOut`-vs-`io.c` au drain,
  bias-zéro-vs-bias-réel). FSM existante inchangée ; compteurs rebouclés entre passes ;
  B-buffer rechargé par slice (fire piloté par S2).
- **Gate** : sim standalone bit-exact vs réplica **inchangé** ; test `bias_add(x,0)==x`
  par dtype (dont `FloatML`) — échec ⇒ fallback bypass ; spill désactivé = historique.
- **✅ VALIDÉE le 19/09/2026** (`MatmulSpillTest` 4/4 + non-régression legacy
  `Matmul/Linear/Conv/BiasAdd/SpillConfig` verte). Écarts et durcissements actés :
  seed **par ligne** entrelacé (pas en bloc — aliasing de fenêtre `temporal`) ;
  entrée de passe verrouillée sur **front `reArm`** (pass-fire explicite, pas de
  départ fantôme sur `tileReady` stale) ; `passDone` pulsé sur le drain de la
  **dernière ligne** (pas par ligne) ; bias-zéro validé I8 + BF16 au triple près
  (fallback bypass **écarté**) ; réplica **zéro-change confirmé** (ordre `fadd`
  identique) ; `LinearLayer` spillée + `passDone` exposé (test smoke : bias une
  fois, `y` silencieux hors passe finale). Discipline bench consignée en tête de
  `MatmulSpillTest.scala` (v3 : valid tenu, un tick/beat, payload pré-posé,
  `ready` armés tôt, feed/collecte entrelacés sur passthrough).

## S2 — Pass-loop `Sequential` + curseur spill `Accelerator` (1-1.5 j)

- `SpillPassController` (1 FSM par couche spillée) : re-fire `reqW`
  (`address = weightsBase + layerOffset + p*sliceBytes`), re-fire A DDR **ou**
  source replay (`StreamTap` snoop passe 0, rejeu passes > 0) ; paire
  `DMAReader`/`DMAWriter` spill + `spillCursor` (reset sur write `0x34`, miroir
  `imgBaseOffset`/`outBaseOffset`).
- Anti-deadlock : passe p+1 après drain+write p terminés ; fencing région Phase 4
  pour le RAW spill inter-passes.
- **Gate** : e2e `BramAdapter` petit + `AxiMemorySim` grand (Linear > capacité déclarée)
  bit-exact vs `ModelReplica` ; assertions contenu région spill par passe ;
  `TILE_CNT`/STOP/curseurs cohérents.
- **✅ VALIDÉE le 20/09/2026** (`SequentialSpillTest` 10/10 + sous-ensembles
  spill/sequential/accelerator/specs verts ; commits `Step S2c`, `Dtep S2c first e2e`,
  puis S2d-1/S2d-2). Ce qui a réellement tourné, écarts et durcissements :
  - **E2E** (`AxiMemorySim`, oracle scala dans le test — la comparaison
    `ModelReplica` complète reste S3) : nœud 0 exclusif P=2 bit-exact **et**
    région = partielles passe 0 (P=1 dégénère, zero-tail prouve seed+bias) ;
    nœud profond P=2 via `StreamTap` (replay on-chip) ; échelle
    auto-identifiante K8/K12/K16 (chaque passe reconnaissable) ;
    K64-P2 et K64-P8 (fit exact 344B, 8 passes) ; rerun back-to-back
    (y/région identiques, `TILE_CNT`/STOP/`MODE`/`0x34` cohérents).
  - **Géométrie slice** : la couche spillée instancie le moteur en
    `[M,Ks]×[Ks,N]` + fenêtre-K sur A (un moteur full-shape starve ses
    compteurs shape-driven sur des flux partiels). Beats hors fenêtre
    acceptés-puis-jetés (l'amont ne stalle jamais), `passIdx` stable en prelude.
  - **`refetchW` retardé d'un cycle** (`preludeHeld`) : le plan de fetch adresse
    depuis le registre `passIdx`, qui settle un cycle après `cnt`.
  - **Bias** : re-arm sur la prelude finale uniquement, gaté `passLast`
    (un pulse à chaque prelude peut charger un bias partiel à cheval sur le
    flip `passLast`) ; zéros hors passe finale côté moteur (S1).
  - **`restartA` ne re-arm PAS le buffer/streamer image** : le re-fire des
    mêmes sweeps se resynchronise seul (ping-pong + `tileReady`) ; re-armer
    coupait le sweep en vol et le repack legacy non-flushable gardait un
    octet résiduel → décalage de trame permanent (K64-P8 faux). Le `reArm`
    image reste sur le front START (nouvelle inférence).
  - **Contrôleur** : pas de sortie de prelude au cycle d'entrée (les flags
    servants y sont encore rassis) — sinon prelude 2 sauté et stall P≥3.
  - **Fetch W de START = slice 0 forcée** : le registre `passIdx` retient
    l'index final du run précédent au 2ᵉ START (RERUN faux sinon).
  - **Re-stream A** : nœud 0 exclusif = re-fire image DDR (gratuit) ; nœud
    partagé/profond = `StreamTap` (snoop passe 0, replay verbatim, budget S0) ;
    jamais de re-fire d'un nœud partagé (dupliquerait dans l'autre branche).
  - **RMW inter-passes** : `WaitFence` sur `writerDone` (B du drain) avant le
    prelude suivant — RAW sur la région unique. La visibilité inter-masters
    repose sur l'ordre du contrôleur mémoire (fence strict `DdrAdapter`,
    Phase 4.1) et reste séquentiellement cohérente sur `AxiMemorySim`.
  - **Contrat runtime `STREAM_PER_PASS` confirmé** : spill × residency interdit
    (`refetchW` supprimé sous residency → stall bruyant) ; `MODE = 0` asserté
    par run en e2e. Une seule couche spillée (v1), `temporal >= 1`, A éligible
    (nœud 0 exclusif ou ≤ budget replay).
  - **Contrat layout DDR W** (découverte e2e majeure) : l'engine consomme la
    région en colonne-major (`readAddr = n*chunksK+k`, côté réplica
    `slice(o*K..)`), donc l'ordre physique est le transposé. Legacy (non-spill) :
    un seul fetch de toute la région, ordre `n*K+k`. Spill : chaque passe fetche
    UNE slice contiguë, donc l'ordre physique doit être **slice-transposé
    contigu** — `p*Ks*N + n*Ks + k_local`, cf. `programmedW(ks)` de
    `SequentialSpillTest`. Un whole-transpose legacy n'est PAS découpable : les
    slices y sont strideés (Ks valeurs par colonne puis trou de K-Ks). P=1
    (Ks=K) dégénère exactement en whole-transpose. **Outillage** :
    `WeightMemoryLayout.buildDeterministicWeights` (test/réplica + CLI `generate`)
    émet encore le whole-transpose legacy ; quand une couche spillera il devra
    émettre le slice-transposé (et le réplica le consommer sans double
    transposition). Tant que ce n'est pas fait, les benches spill programment
    la DDR elles-mêmes.
  - **Bring-up sim** : settle des agents mémoire avant stimulus (un B/R
    parasite au release du reset cale un compteur DMA) ; gardes saturantes
    `pendingB`/`burstRemain` (neutres au formel) ; discipline bench v3
    (le VCD `withWave` est lossy, le bench fait foi).

## S3 — Formel + non-régression + docs (0.5-1 j)

- **✅ Formel contrôleur (20/09/2026)** : `SpillPassControllerFormal` (safety,
  BMC 40, Boolector, 16.5 s) + `SpillPassControllerLivenessFormal` (liveness
  bornée, BMC 80, 12.1 s) — progression 0→P-1, prelude jamais sauté (fix S2d
  pinné), commandes/pulses one-shot, `refetchW` tenu jusqu'à acceptation
  (drop uniquement fire/residency), terminaison sous inputs fair, covers de
  non-vacuité. « Bias exactement N beats » déjà couvert par `BiasAddFormal` ;
  bias-une-fois prouvé au niveau contrôleur (le gate final-only de
  `Sequential` reste prouvé par l'e2e zero-tail).
- **✅ Non-régression complète (20/09/2026)** : `test-all` + `test-all-formal`
  (dont les 2 harnais ci-dessus) + `test-all-python` verts.
- **✅ PR « réplica spill » — close la S3 (voir commits R1-R6, branche `ddrImpl2`)** :
  - outillage `WeightMemoryLayout` slice-transposé (R1) : région W émise en
    `p*Ks*N + n*Ks + k_local`, P=1 dégénère en whole-transpose legacy ;
    `LayerWeightInfo` porte `spillKSlice`/`spillPasses` (défauts = legacy) ;
  - fold réplica `spillWidth` (R2) : `LayerReplicas.linear(..., spillKSlice=-1
    = legacy, instruction par instruction)`, `DenseHandlers` regroupe les
    lignes logiques depuis l'ordre physique + rejette tout mismatch
    couche/layout (pas de double transposition silencieuse) ; int et BF16
    bit-exacts vs oracle dense sur le même modèle logique ;
  - comparaison `ModelReplica` complète sur HW spillé (R3) :
    `SequentialReplicaSpillTest` I8-K8-P2 et BF16-K8-P2, `dev=0.0`, cohérence
    `TILE_CNT`/STOP/`MODE`/`0x34` ;
  - CLI `test` minimale (R4) : le scaffold propage `dut.memory`
    (fini les bases en dur) + timeout ×4 sur modèle spillé ;
    `tests/universal/UniversalSpillDemo.scala` prouve `spinalml test`
    bit-exact sur spill ; seam `memorySimConfig` (défaut = modèle idéal
    historique) où le futur `--stress` branchera la pression timing sans
    toucher l'oracle ;
  - trafic/inférence mesuré (R5, sondes AR/AW en sim, modèle
    `P×(W_slice+A+2·M·N)`, M=1 N=4 K=8 Ks=4 P=2) :
    - I8 : 93 cycles, AR = 8 beats (64 B lus : W 2×16 + A 2×8 + bias + seed),
      AW = 1 beat pré-collecte (drain spill 4 B ; writeback out hors fenêtre
      de collecte), région W = 40 B ;
    - BF16 : 93 cycles, AR = 14 beats (112 B : W 2×32 + A 2×16 + bias + seed),
      AW = 1 beat (drain spill 8 B), région W = 72 B.
- **Docs** : `ddr_impl.md` §5.3 + `ddr_replica_status.md` §7 à jour (S2d-3 +
  PR réplica) ; la présente note est la clôture S3 définitive.
- **Gate du cap** : `test-all` + `test-all-formal` verts (R6 ; python skippé
  par consigne).

## Volontairement hors v1 (v2+)

A-spill vers DDR (gros A profonds), Conv spillées, N-split (K petit / N énorme),
KV-cache, folding L2, spill × residency/prefetch, `parallelN` + spill.
Chacun se branche sur le gabarit S0-S2 (pass-controller, curseurs, fit-check) sans rework.

Le cadre général de scaling (tuiles physiques LUT/DSP vs slices informationnels
BRAM/trafic sur les axes M/K/N, modèle de ressources, planner budgété board)
est esquissé dans `docs/tiling_resource_scaling.md` — c'est la généralisation
du K-slice validé ici, et le prérequis du cœur plié (Phase 3).

## Stratégie PR

S0+S1 sur une branche (commits = saves) ; scission S2+S3 en 2ᵉ PR seulement si le
diff S2 explose. Couture : S0+S1 = `LayerSpec`/`MatmulOp`/unitaires ;
S2+S3 = `Sequential`/`Accelerator`/e2e+formel.
