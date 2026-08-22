# WIP — wXaY Attention & corrections de précision (note de transfert)

> **Note temporaire** (à supprimer une fois intégrée). Résume l'état du travail
> sur la quantification weight-only (wXaY) des attentions et les bugs de
> précision découverts à cette occasion. La partie **Linear** (wXaY + scaled
> `Cast` + sweeps) est terminée et validée : voir `docs/opsSupport.md`,
> `docs/opsDocs.md`, `docs/roadmap.md`.

## 1. Ce qui a été livré côté attention

- `ClassicalAttentionHW` accepte désormais `weightType: HardType[TW]` +
  `weightScales: Seq[Double]` (scales partagées Q/K/V/O, longueur 1 = per-tensor
  ou `embedDim` = per-channel par colonne de poids).
- Déquantisation via un `cast` scalé inséré **une fois par matrice à la
  frontière io**, avant les `StreamFork` de têtes (aval inchangé).
- Condition legacy : déquant uniquement si `(FloatML, SInt)` — tout le reste
  garde le câblage direct d'origine.
- Golden models Python : `classical_attention_hw_wxay` /
  `multi_head_attention_hw_wxay` (désérialisation via `dequant_hw`, réutilise
  la chaîne existante).
- Tests Scala : `AttentionQuantTestComp` / `MultiHeadQuantTestComp`
  (matrice de compilation 6 combos `{I8,I4}×{BF16,FP8_E4M3,FP4_E2M1}` +
  PerChannel, dans les suites `ClassicalAttentionTest` / `MultiHeadAttentionTest`,
  package `spinalML.layers` malgré le dossier `attention/`).
- Tests cocotb : runners `_wxay` dans `test_classicalattention.py` /
  `test_multiheadattention.py` (toplevels `AttentionQuantTestComp` /
  `MultiHeadQuantTestComp`, filtres `-z` sur les noms `wXaY`, uniques —
  leçon du collision `w8a16`/PerChannel de Linear appliquée).

## 2. La grosse correction de précision (`utils/Float.scala`)

Trois bugs trouvés dans les unités float, tous masqués jusque-là :

1. **Wrap exponentiel** : `expSumSInt` (mul) et `newExpSInt` (add) calculés sur
   `expBits+1` bits signés sans expansion → au-delà du max représentable,
   l'exposant wrappait en négatif → sortie **zéro au lieu de saturer vers ±inf**
   (ex. FP4 : `3.0×3.0 = 0`). Fix : élargissement à `expBits+3`.
2. **Troncature au lieu d'arrondi** : `Float.add` et `Float.mul` jetaient les
   guard/sticky bits (`finalMantissa = normalized >> k`). Absorption classique :
   `1.0 + 0.0074 → 1.0` au lieu de `1.0078125` → softmax non normalisé.
   Fix : **round-to-nearest-even** (guard && (sticky || LSB)) avec carry
   d'arrondi sur l'exposant. ⚠️ Piège SpinalHDL rencontré : `Bool.asSInt`
   vaut **−1** quand true → passer par `.asUInt.intoSInt.resized`.
3. Goldens synchronisés dans le même mouvement :
   - Python : `floatml_mul` / `floatml_add` (round-to-nearest),
   - Scala : `FloatGolden` (oracle du sweep, `FloatSweepTest.scala`),
   - nouveau balayage systématique near-1.0 (« absorption ») : ~3,5M paires
     BF16 exhaustives sur `add(a∈[1..4], b∈petits)` — c'est le coin qui avait
     échappé à l'échantillonnage aléatoire.

Leçon méthodologique conservée dans `docs/symbolicTest.md` : les oracles
auto-comparés ne certifient que la fidélité de netlist ; la justesse math est
gardée par les sweeps (`FloatSweepTest`) jusqu'aux oracles formels indépendants.

### ⚠️ À revérifier (non rejoué depuis le changement d'arrondi)
Les tests Python `test_cast.py`, `test_matmul.py`, `test_softmax.py` étaient
verts AVANT le passage en round-to-nearest. Les goldens ont été synchronisés
mais ces suites doivent être relancées pour confirmer le bit-exact.

## 3. Bug structurel corrigé au passage : contexte attention = P@Vᵀ

La sortie de la matmul V est streamée **row-major** ; le `repack` groupait donc
des **lignes** de V en beats, alors que le B-buffer de la matmul consomme des
**colonnes** (convention column-major des poids). L'attention calculait donc
`P @ Vᵀ` au lieu de `P @ V`. Masqué jusque-là par les tolérances relaxées
(softmax PWL) et les petites valeurs aléatoires des tests uniformes.

Fix dans `ClassicalAttentionHW` : `transpose(v)` avant le fifo/repack, avec
re-déclaration de l'orientation logique `[seqLen, headDim]` (sinon échec de
shape en multi-têtes). `TransposeOp` vérifié propre en mode 2 tuiles.

## 4. 🚨 Problème OUVERT : ligne fantôme du Softmax1D en mode batched

**Symptôme** : pour toute tuile N ≥ 1 (2ᵉ inférence et plus dans le même
composant), la **ligne 0** des probs consomme une duplication de la **dernière
ligne de probs de la tuile précédente** → toutes les lignes sont décalées d'un
rang → Y faux. La ligne 1 reste ≈ correcte.

**Preuve** (trial 1, w8a16, données seed 42) : `Y_hw[0] = [-0.484, 0.672]`
correspond exactement à `P_stale @ V_tile1 @ Wo` (= `[0.672, −0.915] @ Wo`),
tandis que le golden attend `[2.203, −0.182]`.

**Ce qui est sain isolément (vérifié par sondes Scala 2 tuiles)** :
- `Softmax1D` seul : 4 rows émises, aucune dupliquée ;
- `ExpOp` BF16 (branche PWL → `UnaryPWLOp`) seul : 4 beats propres ;
- `TransposeOp` seul : 2 tuiles parfaites.

**Hypothèse de travail** : interaction **stall/reprise** — quand le
consommateur aval (A-input de la matmul de contexte, qui attend d'abord son
B-buffer plein) met `y.ready` bas à la frontière de tuile, un étage du pipeline
softmax (join final `invSum × carryExp` ou un `m2sPipe` interne) rejoue sa
dernière donnée au redémarrage. Le beat fantôme reste pendi à l'entrée A de la
matmul de contexte et est consommé en tête de tuile suivante.

**Piste non explorée** : bisect par suppression progressive d'étages dans une
réplication fidèle du corps de `Softmax1D` (sans forks d'observation — ils
modifiaient le comportement), alimentée par un consommateur volontairement lent
pour reproduire le stall. Les sondes temporaires utilisées ont été supprimées ;
les motifs utiles restent décrits ici (moniteurs `fork { waitSampling(); if
(valid && ready) ... }`, séquençage strict inter-tuiles, dumps via
`log_math_line` / `--debug-math` → `true_math_errors.log`).

**Contournement actuel** : aucun. Les tests quant attention valident le
**trial 0** (écarts ≤ 0.016 vs golden) ; les trials ≥ 1 échouent sur la ligne 0
(`err > 0.25`). Ce n'est lié à **aucune quantification particulière** — le bug
est préexistant et touche potentiellement aussi les chemins uniformes multi-
tuiles (masqué par les tolérances et les petites valeurs).

## 5. État des tests au moment de la note

| Suite | État |
|---|---|
| Scala complet (`mill spinalML.test`) | ✅ 242/242 |
| Python `test_linear.py` (dont 7 quant wXaY) | ✅ |
| Python `test_classicalattention.py` / `test_multiheadattention.py` quant | ⚠️ trial 0 ✅, trials ≥ 1 ❌ (bug §4) |
| Python `test_cast/matmul/softmax` | ⚠️ à relancer post-round-to-nearest (§2) |
| Python attention **uniformes** | ⚠️ à relancer post-fixes (non rejouées) |

## 6. Fichiers touchés (session attention + précision)

- `spinalML/src/spinalML/utils/Float.scala` (wrap + arrondi + `fromDouble`)
- `spinalML/src/spinalML/attention/ClassicalAttention.scala` (wXaY + fix P@V)
- `spinalML/src/spinalML/layers/Linear.scala`, `ops/cast.scala` (session Linear)
- Tests : `FloatMathTest` (anti-wrap), `FloatSweepTest` (nouveau), `LinearTest`,
  `CastTest`, `AttentionTest.scala`, `symbolicTest/layers/LinearFormal.scala`
- Python : `golden_models/ops.py`, `golden_models/dtypes.py` (inchangé),
  `test_linear.py`, `test_cast.py`, `test_classicalattention.py`,
  `test_multiheadattention.py`
- Docs : `opsSupport.md`, `opsDocs.md`, `roadmap.md`, `symbolicTest.md`

## 7. Feuille de route pour la reprise

1. Corriger le replay après stall dans `Softmax1D` (§4) — sonde deux-tuiles
   recommandée : comp réel + consommateur lent + compteur de beats Y.
2. Relancer `test_cast/matmul/softmax` puis les attentions **uniformes**.
3. Relancer les suites quant attention : objectif = 7/7 classical + 7/7 multihead.
4. Suite Scala complète + CI verte → PR (initiée par un humain, cf. README).
