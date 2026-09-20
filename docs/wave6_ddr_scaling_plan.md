# Wave 6 — Cap DRAM : du one-shot on-chip au scaling DDR

> Statut : plan de cap validé (2026-09-16). Document le plus important de la
> séquence : il fixe la direction après les arrondis (Wave 4/5, voir
> `docs/wave4_wave5_plan.md`).

## 0. Objectif

Lever le plafond actuel : **`Sequential` est en contrat one-shot** — chaque
buffer contient un tenseur entier, chaque `start` refetch image + tous les poids
depuis la DDR et exécute une inférence complète. Cela plafonne silencieusement
les modèles à ce qui tient on-chip. Cible Wave 6 :

1. des **modèles plus gros que la BRAM** (poids et/ou activations spillés en DDR) ;
2. du **flux continu** (frames vidéo, tokens LLM) avec résidence/prefetch ;
3. le **folding L2** (une couche physique, poids rechargés par passe).

Métrique de succès du cap : faire tourner sur Tang Primer 20K (GW2A-18, 27 MHz)
un modèle **strictement plus gros que la BRAM** (ex. MLP 4k ou conv multi-canaux)
en **flux continu**, poids résidents entre passes, avec bande passante DDR et
latence mesurées et documentées.

## 1. État de départ

### Acquis (validés, dont une partie hors CI)

| Brique | Statut | Preuve |
|---|---|---|
| `DdrAdapter` écriture + arbitrage hôte/accélérateur | ✅ Wave 2 (MEM-04b) | `test_ddr_adapter_accel_write` |
| Adapters mémoire synth-clean (1 port d'écriture, masque byte) | ✅ Fix 24 | BSRAM 27/46, bitstream flashé + inférence |
| Weight residency (Phase-2a, CSR `0x10` bit0) | ✅ | `WeightResidentChainTest` (AR=0 en steady) |
| Prefetch (Phase-2b, CSR `0x10` bit1 + `0x14` RELOAD) | ✅ | `WeightPrefetchChainTest` (eager bit-exact) |
| Tiling vertical V1 + halo (`tileHeight`) | ✅ | `BandTilingTest`, `WideResidualTilingTest` |
| Contrôle continu RUN/STOP (`0x1C`), `TILE_CNT` (`0x18`) | ✅ | `MnistContinuousTest` |
| Re-arming inter-start (résidence-friendly) | ✅ | `MnistChainedTest` (10 inférences bit-exactes) |
| `DMAReader2D` patches 2D (tuiles, lignes non alignées) | ✅ | formels + tests dédiés |

### Dette critique

Les seules suites end-to-end couvrant la résidence, le prefetch, le tiling et
le continu sont **archivées et hors CI** (`archive/test/examples/`) :
`WeightResidentChainTest`, `WeightPrefetchChainTest`, `MnistChainedTest`,
`MnistContinuousTest`, `BandTilingTest`, `SdbSwapTb`. Conséquence : **les
chemins DRAM/scaling ne sont plus vérifiés par `test-all`** — on ne sait pas si
les waves 1-3 ont régressé quoi que ce soit sur ce périmètre. C'est le risque
n°1 et la priorité 0.

### Ouvert (roadmap §4/§5, à traiter dans les priorités ci-dessous)

- **Advanced Tiling (Matrix A)** : spill des sommes partielles M×N en DDR.
- **DAG tap contract** sous flux continu (FIFOs dimensionnées une tuile).
- **Resource-scaling V1** : lanes figées (`Linear → inFeatures`, `Conv2D → K²`),
  arbiter read single-stage, contrat du dernier beat partiel.
- **Layer Folding L2** (poids re-fetchés par passe).
- **Partition mémoire configurable** (fin de `imgBase = 0x10000` /
  `weightBase = 0x20000` figés) et bring-up DDR3 réel.

## 2. Priorité 0 — Filet de sécurité (1–2 j)

**Objectif : aucun travail de scaling sans tests vivants.**

1. **Restaurer les suites archivées** dans
   `spinalML/test/src/spinalML/examples/` (ou les porter vers
   `tests/python/test_accelerator.py` avec un top à poids + écritures CSR
   `0x10`/`0x14`), et les brancher dans `test-all` :
   `WeightResidentChainTest`, `WeightPrefetchChainTest`, `MnistChainedTest`,
   `MnistContinuousTest`, `BandTilingTest`, `SdbSwapTb`.
2. **Créer un e2e DDR dédié** : top à poids utilisant `DdrAdapter` +
   `AxiMemorySim` ; vérifier les motifs AXI AR (résidence = 0 AR en steady,
   prefetch eager = 0 AR dans la fenêtre START→premier beat), le write-back et
   le reload.
3. **Smoke test synthèse en CI** (garde-fou Fix 24) : build Mnist/tang avec
   `--synth-only` ; échec si `using FF mapping`, BSRAM > 46, ou Fmax < cible.
4. **Corriger les références périmées** (`open-mysteries.md`, `full_roadmap.md`
   §8, `test-budget.md`) qui citent ces suites comme actives.

**Gate de sortie** : ces suites tournent en CI (Scala ou Python) ; un smoke
synthèse vert ; plus aucune référence doc périmée.

## 3. Priorité 1 — Advanced Tiling (Matrix A) (2–4 j)

**Objectif** : matrices dont les sommes partielles M×N ne tiennent pas dans les
accumulateurs — spill vers la DDR entre passes d'accumulation.

1. **Contrat de spill** : layout DDR des tuiles d'accumulateurs, alignement
   beats, taille max par bande.
2. **Write-Back / Read-Modify-Write** via le maître AXI (désormais writable) :
   accumulation par passes, RNE + saturation à la requantification finale.
3. **Traçage de complétion par tuile** : bias/activation appliqués **une seule
   fois**, sur la passe finale.
4. **Validation** : réplique bit-exacte sur matrice > capacité des
   accumulateurs ; formel du contrôle multi-passes ; budget LUT/BSRAM consigné.

**Risque** : bande passante DDR à 27 MHz (64 bits) — mesurer tôt le coût du
spill ; si prohibitif, réduire les passes (tiling plus fin) ou viser une DDR
plus rapide.

## 4. Priorité 2 — DAG tap sous flux continu (1–2 j)

`TapBuffer` dimensionne ses FIFOs pour **une tuile** (garantie : la branche
différée draine dans l'inférence). En flux continu, le producteur pousse pendant
que la branche différée travaille.

Options à trancher (coût LUT vs latence) :
- **capacité multi-tile** pour les taps,
- **règle d'admission** (la branche différée doit rattraper en ≤ N tuiles —
  vérifiée à l'élaboration),
- **spill-DDR des taps** pour les skip connections.

**Pré-requis** : fermer M1 (`open-mysteries.md`, gearbox flushable × DAG) —
suspect restant = sensibilité de pacing d'un op consommateur ; bloquant pour la
Phase 2/3 selon le doc. Tests : chaîne WideResidual-like en flux continu (pas
seulement inférence par inférence).

## 5. Priorité 3 — Resource-scaling V1 (2–3 j)

Trois décisions V1 documentées (audit août 2026) devenues bloquantes pour les
gros modèles :

1. **Streaming des poids à largeur fixe** : `Sequential` dimensionne les lanes
   `Conv2D → K²`, `Linear → inFeatures`, attention → `embedDim` — un 7×7 coûte
   49 lanes, un Linear 4096 en instancie 4096. Passer à un datapath à largeur
   fixe avec streaming des poids (knobs par couche, cf. `weightLanes`).
2. **Arbiter AXI read en cascade** : aujourd'hui `Axi4ReadOnlyArbiter`
   single-stage ; les DAG multi-couches exigent un arbre.
3. **Contrat du dernier beat partiel** : `Tensor` n'exige pas
   `totalElements % lanes == 0` (beat final partiel, ex. Conv2D [2][5][5]=50 à
   lanes=4) — à préserver dans tout refactor.

**Gate** : un modèle du type « 7×7 conv / Linear 4096 » tient dans le budget
d'une board cible, régression bit-exacte sur les modèles existants.

**Généralisation 3 axes** : ce qui précède est la version V1 (lanes, arbiter,
contrat du beat partiel). Le cadre complet — tuiles physiques (LUT/DSP) vs
slices informationnels (BRAM/trafic) sur les axes M/K/N, modèle de ressources,
policy globale + résolution par couche et planner budgété board — est documenté
dans `docs/tiling_resource_scaling.md` (phases R0-R3).

## 6. Priorité 4 — Layer Folding L2 (LLM-style) (2–4 j)

Une **seule couche physique**, poids **re-fetchés par passe** (au lieu d'être
résidents) : c'est ainsi qu'un petit FPGA fait tourner un LLM. Pré-requis :
prefetch 2b + contrôle continu + (idéalement) advanced tiling. Le contrôleur
L1 (wrap mux + compteur) est réutilisé ; seule la source des poids devient
par itération. **Gate** : chaîne transformeur miniature bit-exacte vs réplique.

## 7. Priorité 5 — Sous-système mémoire & bring-up DDR3 (3–5 j + hardware)

1. **Partition mémoire configurable** : généraliser la formule figée
   (`AxiReadMem`, `imgBase`/`weightBase`) en contrôleur multi-régions
   configurable (Refactoring 1.3 : abstraction `MemoryAdapter` complète,
   HyperRAM/PSRAM/DDR/ASIC SRAM).
2. **DdrAdapter sur la DDR3 physique du Tang Primer 20K** : contraintes,
   horloge, calibration ; lancer un modèle **> BRAM** sur silicium.
3. **Mesures** : bande passante DDR effective, latence par inférence, ressources
   (LUT/FF/BSRAM/DSP), Fmax ; consigner dans le README de l'exemple.
4. Optionnel : runner hôte (type Radxa) pour de gros vecteurs d'entrée.

## 8. Ordre, dépendances, risques

| Ordre | Chantier | Dépend de | Durée indicative |
|---|---|---|---|
| 0 | Filet de sécurité (tests + smoke) | — | 1–2 j |
| 1 | Advanced Tiling (A) | P0 | 2–4 j |
| 2 | DAG tap continu | P0 (+ M1 fermé) | 1–2 j |
| 3 | Resource-scaling V1 | P0 (peut avancer en parallèle de P1/P2) | 2–3 j |
| 4 | Folding L2 | P1/P2 + prefetch | 2–4 j |
| 5 | Mémoire configurable + DDR3 réel | transversal (démarrer tôt pour le hardware) | 3–5 j |

**Risques principaux** :
- **Bande passante DDR** : 27 MHz × 64 bits ≈ 216 Mo/s théoriques — le spill
  d'accumulateurs doit être mesuré avant d'être généralisé.
- **Calibration DDR3** et contraintes de timing sur Tang : à isoler tôt.
- **M1** (gearbox × DAG) non fermé = risque de pacing sur tout flux continu.
- **Temps de test** : `WideResidualTilingTest` ≈ 53 min ; prévoir des variantes
  réduites en CI et un run long hors CI.
- **Explosion des lanes** (P3) : ne pas refactorer sans régression bit-exacte.
- **Dette de test** : chaque nouvelle brique doit arriver avec son e2e en CI,
  sinon on reconstitue le problème de la P0.
