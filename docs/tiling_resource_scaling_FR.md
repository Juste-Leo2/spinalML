# Tuilage & Scaling Ressources — Anticipation de Design

> **Note perso** : traduction française de `tiling_resource_scaling.md`,
> uniquement pour Léo — volontairement **non liée** à la documentation
> principale (aucun renvoi depuis `roadmap.md`/`full_roadmap.md`/etc.).
> Si le contenu évolue, mettre à jour les deux versions.

> **Statut** : note de design pour le futur (rien ici n'est implémenté au-delà
> de l'axe K). Fige le vocabulaire tuile/slice et la direction du
> resource-scaling, actés pendant la clôture du spill compute-side S2/S3
> (sept. 2026).
> **Public** : quiconque touche au dimensionnement de `LayerSpec` /
> `MatmulOp` / `Sequential`, au refactor cœur-plié (Phase 3), ou au plumbing
> board/planner.
> **Docs liés** : `ddr_final_impl.md` (spill compute-side S0-S3),
> `ddr_impl.md` §5.3 (spill générique), `wave6_ddr_scaling_plan.md` P1/P3,
> `full_roadmap.md` §5, `roadmap_board.md` §4, `roadmap.md` pilier 5.

## 1. Pourquoi ça existe

Aujourd'hui chaque couche instancie son propre datapath dimensionné *par sa
forme* (`Sequential` dimensionne les lanes du DMA de poids en
`Linear -> inFeatures`, `Conv2D -> K²`, attention -> `embedDim`). Un Linear
à 4096 entrées coûte alors 4096 lanes parallèles. C'est acceptable à l'échelle
MNIST et bloquant pour tout le reste : un FPGA edge ou un ASIC ont besoin
d'opérations **moulées dans un budget de ressources**, avec des poids streamés
dans un datapath à largeur fixe.

Le modèle tuile/slice est le vocabulaire et le jeu de knobs qui rend ça
possible : il sépare la **forme matérielle** (LUT/DSP) du **découpage des
données** (BRAM/trafic DDR), pour qu'un modèle soit projeté sur un budget board
au lieu d'imposer sa surface silicium couche par couche.

C'est un **prérequis du cœur plié Phase 3** (`roadmap.md` pilier 1) : un seul
moteur physique time-multiplexé entre les couches exige (a) une forme fixe,
(b) des descripteurs par couche, (c) un planner qui prouve que tout le modèle
tient sur la board.

## 2. Vocabulaire

### 2.1 Spatial vs temporel

- **Spatial** : on duplique le matériel. Plus de multiplicateurs -> plus de
  **LUT/DSP**, moins de cycles. (Aujourd'hui : `lanes`, `parallelN`.)
- **Temporel** : on réutilise le même matériel sur les itérations. **LUT
  plates**, plus de cycles, working set plus petit. (Aujourd'hui : boucles
  N/M/K.)

### 2.2 Trois niveaux de granularité par axe

Chaque axe GEMM (M = lignes, K = profondeur, N = colonnes) se décrit à trois
niveaux :

| Niveau | Sens | Coût principal | Aujourd'hui (axe K) |
| :--- | :--- | :--- | :--- |
| **Chunk** | éléments traités par cycle et par unité MAC | LUT/DSP + framing beat AXI | `lanes` (= `weightLanes`) |
| **Tuile** | matériel instancié pour une itération | **LUT/DSP** | fixe (1 colonne x `lanes`) |
| **Slice** | bloc de données chargé par passe/boucle | **BRAM / trafic DDR** | `spillKSlice` |

Composition par axe : `axe = slice x sous-itérations` et `slice = q x tuile`
(q >= 1). Les buffers se dimensionnent sur les **slices**, les multiplicateurs
sur les **tuiles**.

### 2.3 Tuile physique vs slice informationnel

Deux choses distinctes, volontairement découplées :

- **Tuile physique** = une forme matérielle. La changer **coûte/rend des
  LUT/DSP**.
- **Slice informationnel** = un bloc de données. Le changer **coûte/rend de la
  pression BRAM et du trafic DDR** (ça ne change jamais les LUT en soi).

Les deux ne coïncident que quand une slice est réalisée spatialement. Dans un
moteur purement temporel, la tuile matérielle est fixe : découper ne déplace
que la mémoire et le trafic. C'est pourquoi, aujourd'hui, `spillKSlice`
change la BRAM et le trafic DDR mais **pas** les LUT.

### 2.4 Exemple chiffré (axe K, existant)

K = 64, `lanes` = 2, I8, beat AXI 64 bits :

| `spillKSlice` | Passes P = K/Ks | Buffer B (Ks x N) | Re-streams A | LUT |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 8 | minimal | 8x | inchangées |
| 32 | 2 | 4x plus gros | 2x | inchangées |
| 64 | 1 (legacy) | maximal | 1x | inchangées |

Slice plus grande => moins de passes et moins de trafic, mais plus de BRAM ;
LUT inchangées.

## 3. Les trois axes

| Axe | Sémantique | Borne (on-chip) | Borne (DDR) | Sortie |
| :--- | :--- | :--- | :--- | :--- |
| **K** (profondeur) | accumulation entre passes ; seul axe exigeant partielles + fencing RAW + bias-une-fois | buffer B `Ks x Ns` | RMW partielles `2*(P-1)*M*N` | native (lignes) |
| **N** (colonnes) | blocs de colonnes indépendants ; pas d'accumulation | buffer B `K x Ns` + largeur d'arbre MAC si spatial | A relue `N/Ns` fois | **stridée** M x Ns par bloc -> assemblage requis |
| **M** (lignes) | blocs de lignes indépendants | accumulateurs `(fenêtre M) x Ns`, région spill `Ms x Ns` | W relu `M/Ms` fois (sauf résidence) | native (lignes contiguës) |

Asymétries clés :

- **K est spécial** : c'est le seul axe de *réduction*. Il possède les
  partielles DDR, le fence inter-passes, la paire seed/drain et le bias
  uniquement en passe finale. Les blocages N/M produisent des résultats
  indépendants et n'ont besoin de rien de tout ça.
- **Le découpage N est l'axe LLM** (`outFeatures` énorme : projections vocab,
  FFN up-proj) où les poids `K x N` ne tiennent pas ; son coût est
  l'assemblage de sortie (segments M x Ns éparpillés dans la frame M x N) et
  les re-lectures de A.
- **Le découpage M est l'axe séquence/batch** : la sortie reste contiguë, le
  slicing de A est trivial, c'est W qui est relu. Il borne la région spill et
  la granularité de la frame de sortie, pas le buffer B.
- **`temporal` est déjà une fenêtre M temporelle** : il borne la table
  d'accumulateurs à `min(temporal, M) x N` slots, pas la région spill.

## 4. Modèle de décomposition & contraintes

Les knobs sont conceptuellement indépendants mais couples par cinq
contraintes :

1. **Divisibilité** : `slice | axe` (sans reste), `tuile | slice`
   (sous-utilisation sinon, `q >= 1`), `lanes | Ks` (contrat d'ordre de fold
   du réplica), et `Ks * Ns * dtypeBytes % beatBytes == 0` (framing AXI).
2. **Buffers** : le buffer de poids doit tenir `Ks x Ns` ; la table
   d'accumulateurs tient `(fenêtre M) x Ns`. Ce sont les slices, pas les
   tuiles, qui dimensionnent les mémoires.
3. **Bande passante** : `K/Ks` (ou `N/Ns`) multiplie les re-lectures de A ;
   `M/Ms` multiplie les re-lectures de W. Petites slices = on échange de la
   BRAM contre du trafic DRAM.
4. **Ordre numérique** : l'ordre de sommation par élément de sortie est un
   **contrat de bit-exactitude** (réplica/`MatmulSpillTest`). L'ordre des
   chunks K et l'ordre des passes sont figés par `lanes` et `Ks` ; le blocage
   M/N ne change pas l'ordre par élément (chaque `y[m,n]` reste indépendant).
5. **Framing** : les slices démarrent alignées beat ; une slice à cheval sur un
   mot partiel est illégale dans le plan de fetch actuel.

Conséquence : le planner traite `(mSlice, kSlice, nSlice, mTile, nTile,
lanes)` comme un **petit système de contraintes**, pas comme des entiers
libres.

## 5. Modèle de ressources (cible : R0)

Par axe, la correspondance knob -> ressource que le projet doit savoir
calculer et rapporter à l'élaboration :

| Knob | Ressource principale | Coût secondaire |
| :--- | :--- | :--- |
| `lanes` / `kChunk` | LUT/DSP (largeur MAC, nombre de `DspMul`) | framing AXI |
| `nTile` | LUT/DSP (`nTile` colonnes MAC) | conflits de bancs, fanout |
| `mTile` | LUT/DSP (`mTile` lignes MAC, Option 2 seulement) | broadcast A/B |
| `mSlice` | région spill `Ms x Ns` (DDR), accumulateurs | W relu `M/Ms` |
| `kSlice` | buffer B `Ks x Ns` (BRAM) | A relue `K/Ks`, RMW partielles |
| `nSlice` | buffer B `Ks x Ns` (BRAM) | A relue `N/Ns` |
| `temporal` | FF/BRAM accumulateurs `min(temporal,M) x Ns` | lignes en vol |

Estimations cycles/trafic :

- cycles compute ~ `M * (K/lanes) * (N/nTile)` (Option 1) ou
  `(M/mTile) * (N/nTile) * (K/lanes)` (Option 2) ;
- trafic DRAM par inférence ~ `K*N` (poids, une fois) + `re-lectures * A` +
  `RMW partielles` (spill K seulement).

Le **lien avec le flux** est exactement ça : **la BRAM borne les slices par le
haut** (working set par passe), **la bande passante DRAM les borne par le bas**
(coût des re-fetch), et la forme de tuile fixe le débit de calcul. Une slice
est « bonne » seulement quand les deux côtés tiennent.

## 6. État actuel — ce qui est implémenté

| Fonctionnalité | Statut | Où |
| :--- | :--- | :--- |
| `lanes` / `weightLanes` (chunk K spatial) | **implémenté** | `nn/LayerSpec.scala` (`effLanes`), `ops/matmul.scala` (tableaux `DspMul`), framing par beat |
| `spillKSlice` (slice K temporelle, P passes) | **implémenté + prouvé formellement** | `LayerSpec.Linear.spillKSlice`, `SpillPassController`, fetch/fenêtre slice dans `Sequential`, `MatmulSpillTest`, notes S0-S3 dans `ddr_final_impl.md` |
| RMW partielles DDR (seed/drain, fence, bias-une-fois) | **implémenté** | `SpillPassController`, paire DMA spill, `Sequential` (`spillIn`/`spillOut`) |
| Contrat layout DDR W (slice-transposé) | **implémenté (benches), outillage en attente** | `SequentialSpillTest.programmedW`, note de clôture §S2 |
| `temporal` (fenêtre M temporelle) | **implémenté** | `ops/matmul.scala:310-326` (table `min(temporal,M) x N`), requis pour le spill |
| `parallelN` (N spatial, mono-bloc) | **implémenté dans le moteur, inutilisé par `Sequential`** | `ops/matmul.scala:132` ; `temporal`/`spill` exigent `parallelN=false` (`matmul.scala:58-61`) |
| `tileHeight` (tuilage bandes d'activations pour images) | **implémenté** | plan image `Sequential`, bandes `DMAReader2D` (une *autre* couche de tuilage : activations, pas axes GEMM) |
| Plumbing mémoire (MemorySpec, CSR 0x34, résidence/prefetch, fit) | **implémenté** | `nn/MemorySpec.scala`, `nn/CsrMap.scala`, `Accelerator.reportFit` |

**Non implémenté** : slicing N, slicing M, tuiles spatiales M/N
configurables, API unifiée de tuilage, reporter de ressources, planner piloté
par la board.

## 7. Architecture prévue

Deux directions candidates (décision **ouverte**) :

- **Option 1 — extension weight-stationary** (continuation naturelle) :
  spatial = `nTile x lanes` ; M et K restent temporels. `nSlice` achète des
  LUT/BRAM, `mSlice` n'achète que de la mémoire. Au plus proche du moteur
  actuel, déjà prouvé.
- **Option 2 — output-stationary 2D** (modèle complet) :
  spatial = `mTile x nTile x lanes` ; K temporel. `mSlice` et `nSlice`
  achètent tous deux des LUT/cycles. C'est un **nouveau moteur** (broadcast
  A/B, accumulation par tuile) et ça rouvre le contrat d'ordre numérique du
  spill.

Le cœur plié Phase 3 (`full_roadmap.md` Phase 3, `roadmap.md` pilier 1) tire
l'architecture dans une direction :

- **une seule forme de cœur**, dimensionnée pour la plus grosse couche (ou
  par classe d'op) ;
- les couches deviennent des **descripteurs** (dims, adresses, nombre de
  slices) ;
- les petites couches sous-utilisées tournent sur le même cœur.

Ça **inverse la propriété des tuiles** : aujourd'hui les tuiles sont une
propriété par couche (un moteur par couche) ; dans le cœur plié la tuile est
une **forme de cœur globale**, et seules les slices restent des champs de
descripteur par couche.

## 8. Design de l'API haut niveau (cible)

Trois étages, pas six paramètres libres par couche :

```
niveau modèle   : TilingPolicy / ResourceBudget  // budgets globaux + défauts
niveau couche   : tiling: Option[Tiling] = None  // overrides explicites seulement
                  Tiling(mTile, nTile, kChunk, mSlice, kSlice, nSlice) // -1 = auto
élaboration     : le planner résout -> resolvedTiling par couche + reportResources
```

- **Le budget est global** (la BRAM/LUT/DSP/bande passante physique de la
  board est un seul pool partagé) ; **les valeurs sont résolues par couche**
  (elles dépendent de la géométrie, du dtype et de l'alignement de la couche).
- L'utilisateur écrit la policy une fois ; les overrides par couche existent
  pour les tests et les couches atypiques ; sinon le planner remplit les
  valeurs.
- `spillKSlice` devient l'alias de compatibilité de `Tiling.kSlice`.
- Dans le monde cœur-plié, les valeurs résolues voyagent comme
  **descripteurs de couche** ; la partie tuile converge vers la forme du cœur.
- L'élaboration doit exposer un `reportResources` (comme `reportFit`) et
  fail-fast sur les combinaisons infaisables.

## 9. Intégration board & planner automatique

Les faits physiques sont **statiques par board** et appartiennent aux
`boards/*.json` (ils y sont déjà, partiellement) :

| Élément | Où aujourd'hui |
| :--- | :--- |
| Comptes LUT/FF/BSRAM/DSP | `boards/<slug>.json` `limits` (parsé par `board.py:154-158`) |
| Mots on-chip + bases d'adresses | `memory.onchip_words`, `memory.onchip_addr` |
| Descripteur DRAM | `memory.ddr` (`present`, `size_bytes`, `dataWidth`, `max_burst`) |

Gaps à combler pour un planner :

1. **BRAM en octets** : `limits.bram` est un *nombre de blocs* (46 pour
   GW2A-18) ; le planner a besoin des octets exploitables (taille de bloc x
   compte, moins une marge) -> ajouter `bram_bytes` (ou les kbits par bloc)
   au JSON board.
2. **Bande passante + latence DRAM effectives** : nécessaires pour borner les
   slices par le bas -> étendre `memory.ddr` avec p.ex.
   `bytes_per_cycle` / `latency_cycles` quand la calibration Phase 5b
   arrivera (estimations en attendant).
3. **Marges de sécurité** = *policy* outillage, pas des faits board : les
   garder en défauts code (surchargeables), séparées de `limits`.

Flux : `spinalml build --board X` charge les faits board, construit la
`TilingPolicy`, lance le planner (modèle + board -> tiling résolu par couche),
élabore, rapporte les ressources et fail-fast quand le modèle ne tient pas —
dans le même esprit que `MemorySpec.reportFit` et `ddr_impl.md` Phase 3.

Le JSON board porte de la **capacité, pas des promesses de perf** :
Fmax/routage dépendent du design ; les chiffres de synthèse pourront plus tard
remplir une section `measured` séparée.

## 10. Phases

| Phase | Livrable | Gate |
| :--- | :--- | :--- |
| **R0** | Modèle de ressources + `reportResources` (pur Scala, sans RTL) : `(mTile,nTile,lanes,Ks,temporal,pipeline) -> LUT/FF/BRAM/DSP/cycles/trafic` | calibré sur 2-3 synthèses Tang, rapporté par couche |
| **R1** | `Tiling` unifié sur `Linear`/`MatmulOp` + `TilingPolicy`, `require`s centralisés, alias `spillKSlice` | les modèles legacy élaborent bit-identique ; combinaisons invalides fail-fast |
| **R2** | Implémentation par axe : N d'abord, puis M (Option 1) ou le moteur tuilé (Option 2) | e2e bit-exact par axe (oracle + réplica), échelle + rerun, formel contrôleur étendu |
| **R3** | Planner piloté board : budget -> tiling par couche, fail-fast, report | `build --board X` refuse les modèles infaisables avec des messages actionnables |

## 11. Questions ouvertes

1. **Option 1 vs Option 2** (extension weight-stationary vs moteur 2D tuilé) :
   impacte la propriété LUT de M et la réécriture du moteur spill.
2. **Cible prioritaire** : vision (YOLO sur Tang, Phase 2/3) vs
   LLM/transformer (slicing N, folding L2) — décide du premier axe à
   implémenter.
3. **Heuristique du planner** : allocation bandwidth-first ou BRAM-first ;
   comment répartir le budget global entre les couches (part de MACs, cible
   de latence, ou uniforme par classe d'op).
4. **Périmètre** : axes de tuilage sur la famille GEMM seulement (`Linear`,
   Conv via im2col, projections d'attention) ou toutes les ops.
5. **Ajouts au JSON board** : `bram_bytes`, `ddr.bytes_per_cycle` /
   `latency_cycles` — schéma + timing (Phase 5b).
6. **Interaction résidence** : spill x résidence/prefetch est interdit
   aujourd'hui (`STREAM_PER_PASS`) ; le planner doit encoder cette contrainte.
7. **Contrat numérique** : chaque nouvel axe doit préserver l'ordre de
   sommation par élément prouvé par le réplica (généralisation de
   `Ks % effLanes == 0`).

## 12. Références

- `ddr_final_impl.md` — spill compute-side S0-S3 (axe K, implémenté).
- `ddr_impl.md` §5.3 — spill générique, contrat runtime, fencing.
- `wave6_ddr_scaling_plan.md` P1 (tuilage avancé) et P3 (resource scaling).
- `full_roadmap.md` §5 (architecture mémoire avancée) et Phase 3.
- `roadmap.md` pilier 1 (cœur plié), pilier 4 (edge SLM).
- `roadmap_board.md` §4 (mémoire par board).
- `nn/LayerSpec.scala`, `ops/matmul.scala`, `nn/Sequential.scala` — sites de
  dimensionnement actuels.
