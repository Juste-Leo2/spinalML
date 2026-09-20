# Réplica & Harness — État des lieux pré-Phase 4

> **Statut** : snapshot en lecture seule, 19 sept. 2026. Ne rien implémenter
> avant décision sur la phase 4.
> **Docs liés** : `docs/ddr_impl.md` (phases 1-5), `docs/wave6_ddr_scaling_plan.md`,
> `docs/full_roadmap.md` §7 (gaps universels connus).
> **Périmètre** : `spinalML/test/.../replica/` (oracle logiciel),
> `spinalML/test/.../harness/` (moteur de sim), commande CLI `test`
> (`cli/spinalml_cli/cli.py`, distincte de `test-all` = ScalaTest/Verilator
> et `test-all-formal` = SymbiYosys).

## 1. Ce que le réplica modélise aujourd'hui (exact)

Entrée : `ModelReplica.forwardWithTrace(layers, inputShape, inputTensor, packed)`
(`replica/ModelReplica.scala:72`), dispatch par `LayerSpec` vers
`replica/handlers/`. Layout poids relu via
`WeightMemoryLayout.buildDeterministicWeights(layers, pipelineDtype, axiConfig)`
(`replica/WeightMemoryLayout.scala:71`) — **layout linéaire beat-aligné,
indépendant des `lanes`** (miroir documenté de la boucle `Sequential`).

Ordre d'accumulation float modélisé (= ordre HW supposé), par famille :

| Couche (float) | Ordre réplica | Ref |
|---|---|---|
| `Linear` | fold K par `weightLanes` : `acc = fadd(acc, tree(chunk))`, bias une fois à la fin | `LayerReplicas.scala:179-183`, largeur lue dans `DenseHandlers.scala:20` (`l.effLanes`) |
| `Conv2D` / `Conv1D` | **fenêtre entière en un seul `tree`**, bias une fois à la fin | `LayerReplicas.scala:33-43` / `:63-73` — **aucun paramètre de lanes** |
| Attention (matmuls internes) | chunk trees séquentiels selon les lanes de chaque matmul (`lanes=1` q/k, `seqLen`, `wLanes=embedDim`) | `AttentionHandlers.scala:42-55,111` |
| Pool/norm/activations | `tree` unique ou séquentiel commutatif | `LayerReplicas.scala` |

Chemins int (`IntTensor`) : boucles plates (`DenseHandlers.scala:41-51`,
`conv2DInt` `LayerReplicas.scala:234-243`), arithmétique entière associative →
**insensibles au chunking**, *si* le wrap/requant n'a lieu qu'une fois à la fin
(`wrap(acc)` dense ; `outWidth` via `biasDtype` côté conv,
`ConvHandlers.scala:43`).

**Angles morts confirmés** (zéro occurrence) : `tileHeight`, `temporal`,
`spill` — absents du réplica **et** du harness. Sans effet aujourd'hui car
bandes/temporal ne changent pas l'ordre des `fadd` (prouvé S2/S3).

## 2. Ce que la phase 4 invaliderait (ou pas)

| Chantier phase 4 | Impact réplica | Pourquoi |
|---|---|---|
| Fencing par région, arbiter cascade, `temporal`, tiling | **Aucun** | Ordonnancement / pacing / drainage — ordre des additions inchangé |
| Lanes uniformes `Linear` | **Aucun** (déjà couvert) | `effLanes` déjà transmis au fold |
| Lanes uniformes `Conv/Attention` | **Obligatoire (float)** | Dès que le HW folde la fenêtre K²·inC autrement qu'en un seul arbre, `conv2D`/`conv1D` divergent. Prévoir `lanes: Int = -1` (= mono-chunk actuel, zéro régression) transmis par `ConvHandlers` |
| Spill multi-passes | **Paramétrage** | Si les passes suivent l'ordre des chunks et stockent les sommes **pleine largeur** avec bias final unique : `weightLanes = spillWidth` suffit (Linear), + `lanes` Conv ci-dessus. Si spill rétréci/wrappé par passe : modélisation par passe requise (lourd) |
| Spill chemins int | **Aucun, sous contrainte** | Assoc. entière OK ssi sommes partielles pleine largeur + wrap final unique — **à figer comme contrainte de conception** |

## 3. Limitations actuelles du harness et du CLI `test`

`UniversalTestHarness.run` (`harness/UniversalTestHarness.scala:49-57`) :
`imgBase`/`weightBase` en dur (`0x10000`/`0x20000`), un seul START, aucun CSR
spill, `timeoutCycles = 50000` fixe, collecte `outStream` uniquement, assert
`dev == 0.0` (`:117-118`).

Scaffold CLI `test` (`cli.py:456-536`) : `new {comp}()` sans args, ne lit que
`modelSpec`/`inputShape`/`globalDataType` — ni `dut.memory`, ni `temporal`,
ni `tileHeight`. Le `require` de `reportFit` remonte déjà tout seul à
l'élaboration (échec clair gratuit), mais les bases custom et le spill ne
sont pas pilotables.

Gaps universels préexistants (rappel `full_roadmap.md` §7, à croiser avec les
modèles de validation phase 4) : Linear int pur sans `Cast`, `AvgPool`,
`Sigmoid`/`Tanh`, `Softmax`, Conv2D int multi-canal, attention, packing
I16/I32.

## 4. Décisions ouvertes (bloquaient le démarrage phase 4 — voir §5-6)

1. **Contrat numérique du spill** : sommes pleine largeur + bias/activation
   une seule fois sur la passe finale (réplica quasi inchangé, recommandé)
   vs spill rétréci (économie BP DDR, réplica par passe + preuves lourdes).
2. **Modèles de validation** : Linear 4096 int + Conv 7×7 (gate P3, gaps
   Linear int pur + Conv2D int multi-canal à combler) vs chaînes float
   agrandies (aucun gap) vs mixte.
3. **Scaffold CLI `test`** : propagation `dut.memory` + log capacité dès la
   phase 4 vs reporté au bring-up (phase 5).

## 5. Garantie de non-régression réplica (phases 1-4 livrées)

Règle appliquée à chaque changement réplica des phases 1-4 : **paramètre
optionnel avec défaut = comportement historique au bit près**, jamais de
réécriture de l'ordre existant.

* `conv2D`/`conv1D` float : `lanes: Int = -1` ; `<= 0` = un seul `tree` sur
  toute la fenêtre = l'ancien code, instruction par instruction
  (`LayerReplicas.scala`). `ConvHandlers` transmet `c.effLanes`, qui vaut
  l'ancienne largeur pleine par défaut (`K*K`, `K*inC`).
* Preuve : suites existantes vertes sans modification d'oracle
  (`MnistTest`, `Conv2DTest`, `AcceleratorTest`, …) + sim dédiée à lanes
  étroites bit-exacte (`SequentialTest` Conv2D BF16 `weightLanes=3`).
* Chemins int : aucune modification (associativité entière, wrap final
  unique — valide tant que le spill reste pleine largeur, §2).
* À ajouter avec le compute-side, même pattern : fold `spillWidth`
  (une ligne par handler, défaut = `effLanes`).

## 6. Compute size — ce qui tient *à l'exécution* aujourd'hui

Distinguer deux tailles (toutes deux en octets exacts, conventions
`MemLayout` : ceil par région + alignement beat) :

* **Footprint déclaré** (tient en DDR/sim, vérifié à l'élaboration par
  `MemorySpec.reportFit`) : `image + totalWeightBytes + out (+ spill)`,
  avec `image = (dtypeBits/8) * inputShape.product` (`Accelerator`),
  `totalWeightBytes` exposé par `Sequential`, `out = totalOutBeats *
  beatBytes`. C'est lui qui « scale en sim » depuis les phases 1-4.
* **Working set d'exécution** (doit tenir on-chip *aujourd'hui*) : par
  couche matmul, table d'accumulateurs `M*N` (`MatmulOp`, legacy) bornée à
  `min(temporal,M)*N` en mode `temporal`, + buffer B `paddedK*N` + fenêtre
  A. Le spill K-pass (compute-side) est ce qui fera passer les sommes
  partiels `M*N` en DDR entre passes — voir design figé dans
  `docs/ddr_impl.md` §5.3.

Règle de lecture : un modèle **élabore et charge** dès que le footprint
tient la capacité déclarée ; il **s'exécute** tant que chaque working set
tient l'on-chip (ou draine vers l'aval en `temporal`). Tout écart lève un
`require` d'élaboration explicite, jamais un débordement silencieux.

## 7. Clôture PR DDR-plumbing + anticipation du compute-side

Cette PR introduit la DDR comme **tuyauterie vérifiée** (pas comme
exécution spillée) : `Target` propagé, `MemoryKind`/`FenceConfig`,
`MemorySpec` + fit check, CSR `0x34` live, arbre AXI, lanes uniformes
Linear/Conv, fencing par chevauchement — chaque brique avec tests sim +
preuves formelles vertes, socle non-régressé.

Le compute-side **était anticipé dès le départ** : `docs/ddr_impl.md` §5.3
fige la soudure (drain temporal → mux `spillOut`, seed `spillIn`, slices W
par offset, re-stream A DDR-résident, bias-zéro hors passe finale,
périmètre v1 = `Linear` à A DDR-résident) et `docs/wave6_ddr_scaling_plan.md`
P1 le chiffrait (2-4 j). Il fera l'objet d'une PR dédiée : contrôleur
multi-passes + RMW + fold réplica + spec formelle du contrôleur.

**MàJ 20/09/2026** : S0-S2 sont livrées et validées e2e
(`SequentialSpillTest` 10/10, cf. note de clôture `docs/ddr_final_impl.md`
§S2 — dont le contrat layout W slice-transposé et le fencing inter-passes).
Le **formel contrôleur** (2 harnais : safety + liveness bornée) et la
**non-régression complète** (`test-all` + `test-all-formal` +
`test-all-python`) sont verts. Reste, **reporté à une PR dédiée « réplica
spill »** : support réplica d'un modèle spillé (fold `spillWidth` si besoin),
outillage `WeightMemoryLayout` slice-transposé (consommé sans double
transposition), comparaison `ModelReplica` e2e, mesure du trafic
`P×(W_slice+A+2·M·N)` et note de clôture S3 définitive.
