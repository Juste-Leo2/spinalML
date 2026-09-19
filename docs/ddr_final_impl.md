# Compute-side spill — Plan final d'implémentation (étapes S0-S3)

> **Statut** : plan figé le 19/09/2026, exécution en cours (S0 d'abord).
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

## S1 — `MatmulOp` slice engine (1 j)

- Ports : `spillIn` (seed, p > 0), `spillOut` (drain, p < P-1, muxé sur `stateEmitRow`),
  `passFirst`/`passLast` (mux zéro-vs-spill à l'init, `spillOut`-vs-`io.c` au drain,
  bias-zéro-vs-bias-réel). FSM existante inchangée ; compteurs rebouclés entre passes ;
  B-buffer rechargé par slice (fire piloté par S2).
- **Gate** : sim standalone bit-exact vs réplica **inchangé** ; test `bias_add(x,0)==x`
  par dtype (dont `FloatML`) — échec ⇒ fallback bypass ; spill désactivé = historique.

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

## S3 — Formel + non-régression + docs (0.5-1 j)

- BMC contrôleur : progression 0→P-1 + terminaison, bias consommé exactement une fois
  (passe finale), pas de deadlock fetch→compute→drain→seed, bias exactement N beats.
- `MemorySpecFormal`/`CsrMap` étendus au `spillBytes` calculé ; vérif réplica zéro-change
  (fallback fold `spillWidth` si besoin) ; trafic/inférence `P×(W_slice+A+2·M·N)` mesuré
  et consigné ; màj `ddr_impl.md` §5.3 + `ddr_replica_status.md` §7.
- **Gate du cap** : `test-all` + `test-all-formal` + python verts.

## Volontairement hors v1 (v2+)

A-spill vers DDR (gros A profonds), Conv spillées, N-split (K petit / N énorme),
KV-cache, folding L2, spill × residency/prefetch, `parallelN` + spill.
Chacun se branche sur le gabarit S0-S2 (pass-controller, curseurs, fit-check) sans rework.

## Stratégie PR

S0+S1 sur une branche (commits = saves) ; scission S2+S3 en 2ᵉ PR seulement si le
diff S2 explose. Couture : S0+S1 = `LayerSpec`/`MatmulOp`/unitaires ;
S2+S3 = `Sequential`/`Accelerator`/e2e+formel.
