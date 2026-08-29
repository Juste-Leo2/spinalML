# LUT Scaling Session — MatMul temporal window (August 2026)

Session dédiée au **scaling LUT** du W4A8 : pourquoi 34 667 LCs estimés malgré une
largeur de lanes réduite, comment on a identifié le vrai goulot, et ce qui a été
mis en place pour le couper. Termine par la découverte d'un bug préexistant
(**M4**) et les pistes pour le corriger.

---

## Contexte

Le benchmark W4A8 (MNIST mixte : Conv I4/I16 + Linear FP8 288→10) a été synthétisé
sous Yosys (`synth_xilinx -nowidelut`, est. LCs) :

```
Estimated number of LCs: 34667
  FDCE  21606   LUT4 19372   LUT6 13142   DSP48E1 25   RAM32M 142 ...
```

Première théorie (fausse) : la largeur. La décomposition `weightLanes` (M2,
commit `4fea0c3`) réduit la largeur des battements du Linear (288→4 lanes),
mais ça ne change **presque rien** : le Linear pesait 544 LC, le vrai mur était
ailleurs.

## Attribution hiérarchique (Yosys `stat` par module)

| Module | LC | FDCE | Cause |
|---|---|---|---|
| **MatmulOp (le CONV, via im2col)** | **31 705** | **19 302** | le mur : 87 % du total |
| MatmulOp_1 (Linear) | 544 | 171 | largeur réduite = petit |
| tout le reste (`DMAReader*`, `MaxPool`, `Im2ColOp`, FIFOs…) | ~2 900 | ~2 100 | raisonnable |

Décortique `MatmulOp` : l'accumulateur n'est pas le problème de **largeur**, c'est
un problème de **stockage des sommes partielles** :

```
val accumulators = Vec(Reg(accType), M * N)   // Table M x N complète, en registres
```

Conv 24×24 → **M = 576 fenêtres × N = 2 canaux × I16 = 18 432 FF** ≈ les 19 302
FDCE observés ; les 18 944 LUT4 = le **mux d'adressage** `getAccIdx(flatIdx)` sur
1152 accumulateurs. Le FSM calcule toutes les rangées **puis** vide la table
(`stateOutput`) : même avec des lanes infinies, aucun gain.

## Le concept : fenêtre temporelle (`temporal`)

La grandeur qui scale = **combien de rangées de sortie sont « en vol »** pendant
que l'arbre d'addition pipeliné (latence `treeLatency`) produit les résultats.
Au lieu d'attendre tout le tenseur, **chaque rangée est drainée dès qu'elle est
complète** (après le flush du pipeline, 3 + `treeLatency` cycles).

- `temporal = 0` (défaut) : lega I — table M×N complète, netlist strictement
  identique (vérifié bit-exact).
- `temporal = W ≥ 1` : table = **min(W, M) × N** registres, index circulaire
  `(row % slots)`, drain par rangée au centre du FSM.
- **L'ordre des `fadd` est inchangé → bit-exact par construction** (le retiming
  du pipeline est indépendant des données).

Compromis : fenêtre < latence ⇒ petit déficit de débit (stalls de pipeline) —
jamais d'erreur fonctionnelle. `temporal ≥ M` = équivalent du comportement hérité.

## Ce qui a été livré

1. **API** — `Accelerator.temporal: Int = 0` → `Sequential.temporal` → propagé
   aux ops à réduction : `Conv2DLayer`/`LinearLayer` → `matmul(..., temporal)`.
   `require(temporal >= 0)`; `require(temporal == 0 || !parallelN)` (fenêtre =
   chemin séquentiel-N en V1 ; `parallelN` non supporté).
2. **MatmulOp** (`ops/matmul.scala`) :
   - chemin séquentiel-N : `accTable`/`accIdxSel` sélectionnés par `temporal` ;
   - FSM fenêtré : `computeN → flush(3+treeLatency) → emitRow(N beats) →
     next` ; slots = `min(temporal, M)` ; index = `(row % slots) × N + n`.
   - le chemin lega I reste textuellement identique (risque zéro).
3. **Tests** — `MNIST_TEMPORAL=<int>` (MnistTest + Mnistw4a8Test) : bit-exact à
   `temporal ∈ {0, 1, 16}` avec `wLanes ∈ {288, 96, 32, 4}` selon le modèle.
4. **Docs** — `docs/test-budget.md` : knob `MNIST_TEMPORAL`.

## Vérification

- W4A8 + Mnist (5 cas chacun) : **déviation 0.0** à temporal 0/1/16.
- `spinalML.layers.*` (Linear/Conv : 23 tests), 8/8 probes, WIDE SKIP 16×16 : verts.
- **Mesure attendue** (à confirmer chez toi) : MatmulOp ~31,7 k LC → ~1–3 k LC,
  total ~3–6 k LCs.

## Commandes utiles

```bash
# Verilog W4A8 avec la fenêtre (le .v le plus récent)
MNIST_TEMPORAL=16 ./mill spinalML.test.testOnly spinalML.examples.Mnistw4a8Test

# Estimation LUT en une commande (le fichier le plus récent)
V=$(ls -t simWorkspace/Accelerator*/rtl/Accelerator.v | head -1); yosys -q -p "read_verilog $V; synth_xilinx -nowidelut; stat -tech xilinx" | grep -E "Estimated|FDCE|LUT4"
```

## M4 — bug découvert au passage : CORRIGÉ (voir le doc prefetch eager)

**Symptôme** : `WeightPrefetchChainTest` et `WeightResidentChainTest` échouent
(post-prefetch/résidence, image #2 corrompue, logits déviés de ~1–2.5).
**Repro** : reproductible sur le commit `4fea0c3` (« Added explicit lines »,
où `Mnistw4a8.defaultModelSpec` porte `weightLanes = 4`) — **antérieur** au
patch temporal. À `weightLanes = 288` l'ensemble était vert (les 20/20 pré-M2).

**Hypothèse à l'époque** : une géométrie de battements « 1 beat = 1
rangée-neurone » durcie dans les compteurs/passes du préchargement eager
(`loadCanAccept`, staging `stagedW`, `reArm`, `trimToElements` de la gearbox)
alors que le Linear 4-lanes émet **72 battements par rangée-neurone**.

**Bisect historique (suites chaînées → knob `MNIST_WLANES`)** :

| weightLanes | chunksK | Résidence/RELOAD | Prefetch eager |
|---|---|---|---|
| 288 | 1 | ✅ | (288-valide, vert sur 288) |
| 96 | 3 | ✅ | — |
| 72 | 4 | ❌ | — |
| 48 | 6 | ❌ | — |
| 32 | 9 | ❌ | — |
| 24 | 12 | ❌ | — |
| 4 | 72 | ❌ | ❌ |

**Résolution (fin août 2026)** : les différentes causes ont été isolées et
corrigées dans une session dédiée — le goulot n'était pas la géométrie du
Linear 4-lanes mais (1) la FIFO de livraison du streamer JAMAIS purgeée en
monde prefetch (`wStreamer.io.reArm := reqW.fire && !prefetchWorldW`), et
(2) le cache de biais du `BiasAddOp` sans frontière (le dernier logit
absorbait le biais de génération précédente). Détails, preuves et re-prod :
`docs/bugs/2026-08-prefetch-eager-stale-fifo-session.md`.

**Corrections accessoires (faites)** : les suites `Chained`/`Continuous`/
`Banded` appelaient `Mnistw4a8Replica.logits(img)` (repli 288-lanes) → passe
en `logitsK(_, 4)` (le fold du modèle). Les suites migrées restantes
(`WeightPrefetchChainTest`, `WeightResidentChainTest`…) utilisent
`logitsK(_, 4)` depuis M2.

---

*Session : août 2026 — commits associés `4fea0c3` (M2 lanes) + `0bab5f7` (M3 temporal). M3 re-réécrit par-dessus la PR `af91bd3` (oct 2026).*
