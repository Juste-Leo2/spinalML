# Spill par op — comblement Linear + généralisation

> **Statut** : plan acté le 21/09/2026, à merger avec `ddrImpl2`.
> **Docs liés** : `docs/ddr_final_impl.md` §S3 (chaîne Linear close, R1-R6),
> `docs/ddr_stress.md` (Phase A chaos, `--stress`), `docs/ddr_replica_status.md`
> (contrat numérique, règle de non-régression), `docs/tiling_resource_scaling.md`
> (cadre M/K/N général).
> **Décision structurante** : on finit l'enveloppe Linear, on généralise le spill
> aux ops compatibles (Conv), et **ensuite seulement** LiteDRAM (backend orthogonal).

## 0. Spill + réplica, ou réplica seul ? — les deux, inséparables

Règle : **le spill HW et son fold réplica forment une unité de preuve indivisible.**

- Spill HW sans fold réplica = pas d'oracle → le bit-exact est improuvable
  (c'était exactement le gap S3 comblé en R1-R3).
- Fold réplica sans spill HW = code mort qui pourrit (jamais exercé, faux sentiment
  de couverture).
- Réplica seul (sans spill) suffit **uniquement** pour les ops qui ne spilleront
  jamais (pointwise, §2) — et celles-là sont déjà couvertes par les handlers.

Corollaire : chaque op spillée arrive avec son knob `LayerSpec`, son câblage
`Sequential`, son fold réplica à défaut legacy, son layout si l'ordre W change,
son e2e bit-exact et son formel. Pas de demi-PR.

## 1. Inventaire des ops (`nn/LayerSpec.scala`) et décision par op

Légende : HW = spill côté hardware, RPL = fold réplica, E2E = preuve bit-exacte.

| Op | Poids DDR | Moteur HW | Verdict |
|---|---|---|---|
| `Linear` | W `K×N` + bias N | `MatmulOp` (K-split) | ✅ **Fait (R1-R6)** — reste P0 ci-dessous |
| `Conv2D` | W `(K²·inC)×outC` + bias | `Conv2DHW` dédié | ✅ **P1 FAIT** (reste cleanup P1-5, voir §3) |
| `Conv1D` | W `(K·inC)×outC` + bias | `Conv1DHW` dédié | ▶️ **P2** : même gabarit que P1, plus petit |
| `ClassicalAttention` | W Q/K/V/proj via DDR | `ClassicalAttentionHW` (matmuls internes) | ⏸️ **P3 (étude)** : scores `seqLen²` on-chip, spill = projections + KV-cache, structure différente |
| `BatchNorm1D`, `LayerNorm1D` | gamma/beta (vecteurs) | pointwise | ❌ Jamais de spill (poids minuscules, streaming) — réplica ✅ existant |
| `MaxPool*`, `AvgPool*` | aucun | pointwise | ❌ Jamais — réplica ✅ existant |
| `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax` | aucun | pointwise | ❌ Jamais — réplica ✅ existant |
| `Cast`, `Requantize`, `Flatten`, `Repack` | aucun | reshapes | ❌ Jamais — réplica ✅ existant |
| `Add`, `Concat` | aucun | DAG | ❌ Jamais — réplica ✅ existant |

Donc « toutes les ops compatibles » = en pratique **Conv2D puis Conv1D**.
Le reste ne spillera jamais par nature (pas de GEMM, pas de mur `K`).

## 2. P0 — Combler l'enveloppe Linear ✅ FAIT (commits P0a-P0d, `ddrimpl3`)

Tous les cas ajoutés aux suites existantes, verts du premier coup
(aucun fix HW — seul le chemin int M=1-only du réplica a demandé une
correction, P0a) :

1. **M>1 sous spill** (tout est M=1 aujourd'hui, HW + réplica) : la boucle `rows`
   de `LayerReplicas.linear` et le reshape `Sequential` en mode spill ne sont
   prouvés nulle part → cas `ReplicaSpillFoldTest` + `SequentialReplicaSpillTest`
   en `Seq(2, K)`.
2. **Nœud profond (tap) + réplica** : le replay `StreamTap` n'est prouvé qu'en
   oracle main (`SequentialSpillTest` TAP-P2) → cas 2 couches
   `[dense, spill-Ks]` avec oracle `ModelReplica`.
3. **P>2 en e2e réplica** : P=3/4/8 existent en oracle main, pas en réplica →
   au moins un cas P=4 `ModelReplica`.
4. **Multi-couches spill + réplica** : couche spillée en position 1+ (offsets de
   régions R1 en théorie, zéro preuve e2e).
5. **Échelle K64 + réplica** : fit exact 344B / 8 passes prouvés en main seulement.
6. **Sweep seeds chaos + light en suite Scala** : un seul seed heavy en suite
   (light couvert que via CLI) → 2-3 seeds, cas light.
7. **Rerun back-to-back + réplica** : `TILE_CNT`/curseurs prouvés en main, pas
   croisés réplica.

Gate P0 : suites ci-dessus vertes + sélection 12+6 (R6) toujours verte.
Après P0, l'enveloppe Linear est close pour de bon — vérifié le 21/09/2026
(P0d : 12 sim + 6 formels verts, voir R6 pour la sélection).

## 3. P1 — Conv2D-spill (le gabarit Linear, axe différent)

Même découpage en commits qu'en R1-R6. Différences connues à l'avance :

- **Axe de slice** : poids aplatis `(K²·inC)×outC` — le slice coupe l'axe aplati
  (même rôle que K), `spillKSlice` diviseur de `K²·inC`, multiple de `effLanes`
  (déjà `weightLanes`, pattern M2 existant `LayerSpec.scala:39-53`).
- **Moteur** : `Conv2DHW` dédié (pas `MatmulOp`) → équivalent S1 à construire
  (seed `spillIn`, drain `spillOut`, bias final unique).
- **A re-streamé** : flux fenêtré im2col à halo porté (`Sequential.scala:35`) —
  le replay/tap doit préserver l'état de halo, c'est le point dur de P1.
- **Layout** : vérifier si l'ordre W consommé exige une permutation (cf. leçon
  slice-transposé S2) ; sinon réutiliser l'outillage R1 tel quel.
- **Réplica** : fold par passes dans `conv2D` (même pattern que `lanes`, défaut
  = mono-chunk historique), bias une fois, int inchangé (associatif).
- **Preuves** : e2e `ModelReplica` I8 + BF16, chaos-heavy, `--stress` CLI,
  formel contrôleur si nouveau contrôleur (ou extension du prouvé).
- **Statut 22/09/2026 : FAIT.** P1-1 knob `SpillableGEMM`+`spillKSlice`,
  P1-2 plumbing générique, P1-3 moteur `Conv2DLayer` spill, P1-4 layout
  slice-transposé + fold réplica, P1-5 e2e `ConvReplicaSpillTest` 5/5
  (P2 I8 814c + BF16 766c bit-exacts, AW = region beats exacts).
  Chemin : deux bugs trouvés et fixés (plan image 2D ignorant les canaux,
  ordre fenêtre réplica `(c,r,k)` vs `(r,k,c)` HW), puis le deadlock seed
  passe-1 (commande région entière devant les beats image en mémoire
  in-order) fixé en S2e : seed en chunks d'1 beat pacés par la gate du
  reader flushable + trim OFF + drain du pad moteur (`spillPadElems`) +
  latch `writerDoneSeen` — voir
  `docs/bugs/2026-09-conv-spill-seed-deadlock-session.md`.
  P1-6 chaos conv 4/4 (heavy+light, I8+BF16, beats identiques à l'idéal),
  P1-7 CLI `UniversalConvSpillDemo` idéal + `--stress` heavy bit-exacts,
  formels contrôleur (+latch) verts. Reste : PR nettoyage P1-5 (tests TMP +
  flags debug, repoussée avant LiteDRAM) + gate complet.

## 4. P2 — Conv1D-spill

P1 en plus petit (axe `K·inC`, moteur `Conv1DHW`). Réutilisation maximale :
même knob, même fold, mêmes suites adaptées.

## 5. P3 — Attention (étude, pas d'implémentation)

Matmuls internes (QK, AV, projections) + scores `seqLen²` + KV-cache : le spill
s'y décline différemment (spill des projections, cache en DDR). À cadrer après
P1-P2, en cohérence avec la roadmap SLM (`roadmap.md` §4). Le réplica attention
(`AttentionHandlers`, fold lanes existant) est le point d'ancrage.

## 6. Ensuite : LiteDRAM

Backend orthogonal (`ExternalDram → DdrAdapter → LiteDRAM`), mêmes suites
bit-exactes, chiffres BW/latence, puis bring-up HW (Phase 5b). Grâce à P0-P2,
l'intégration ne sera qu'une question de timing, plus de correction.
