# Revue PR `round` vs `main` — 2026-09-18

**Périmètre** : `main...round` (merge-base `3586484`), tip `210c800`.
93 fichiers, +6886 / −2603. Méthode : revue manuelle (sources cœur, CLI) +
2 agents de revue (tests Scala, tests Python/exemples). Les tests de
vérification tournent en parallèle ; les régressions éventuelles seront
signalées séparément.

Statut : `[V]` = vérifié personnellement (code relu et/ou exécution),
`[A]` = remonté par un agent, non re-vérifié.

---

## 1. Bloquants

### 1.1 `Float.toSInt` : arrondi RNE faux pour `0 < |x| < 0.5` `[V]` — **CORRIGÉ**

> **Correctif (2026-09-18)** : gate `!tooSmall` sur l'incrément RNE
> (`Float.scala`, `tooSmall = shiftS < -(mantBits+1)`) + golden Python aligné
> (`ops.py`). Repro d'abord : `F2I (0, 125, 64) [rne]: got 1 instead of 0`
> (`test-all-python -k cast_float` avec golden corrigé, RTL encore bugué),
> puis vert après correctif ; vecteurs ajoutés à `CastTest.scala` et
> `test_cast_float.py`. `test-all -k CastTest` PASS, `test-all-python -k
> cast_float` PASS. Détail ci-dessous (conservé comme trace).

**Diagnostic (avant correctif) :**

- **Fichier** : `spinalML/src/spinalML/utils/Float.scala:497-524`
  (et golden miroir `tests/python/golden_models/ops.py:1244-1251`).
- **Nature** : nouvelle fonctionnalité de cette PR (`toSInt` n'existe pas sur
  `main`), utilisée par `CastOp` FloatML → SInt (`ops/cast.scala:105-109`).
- **Cause** : `dropC` est clampé à `mantBits + 1` (lignes 503-505) pour garder
  les sélecteurs dynamiques en range, mais le commentaire « over-clamping only
  zeroes an already-zero result » (l. 500-501) est **faux** : quand le point
  binaire réel est plus à droite que le clamp, `guard` devient le bit caché et
  `sticky` la mantisse, donc l'incrément RNE (l. 522-524) arrondit à 1 des
  valeurs < 0.5.
- **Contre-exemple mesuré** (golden exécuté, idem RTL) :

  | Entrée BF16 | RNE attendu | Obtenu |
  |---|---|---|
  | 0.25 | 0 | 0 (OK) |
  | 0.375 | 0 | **1** |
  | 0.3 | 0 | **1** |
  | 0.5 | 0 | 0 (OK) |
  | 0.75 | 1 | 1 (OK) |
  | −0.375 | 0 | **−1** |

- **Couverture** : les vecteurs `test_cast_float.py:72-77` (43.5, 1152, −43.5,
  −128, 0, 4, 0.5) évitent tous la zone `(0, 0.5)` mantisse non nulle → vert à
  tort. Même repro Scala `CastTest`.
- **Correctif appliqué** : `val tooSmall = shiftS < S(-(mantBits + 1), eW + 1 bits)`
  puis `roundUp = rne && guard && (sticky || kept.lsb) && !tooSmall`
  (`Float.scala`), golden Python aligné (`tiny = -shift > im + 1`), vecteurs
  ajoutés aux deux suites.

### 1.2 `flash` : sélection du bitstream non bornée au board `[V]` — **accepté par design**

- **Fichier** : `cli/spinalml_cli/flash_runner.py:58-75`.
- `hw_build_root.rglob(bitstream_name)` cherche **récursivement dans tout
  `hw_build/`** (tous boards, plus les artefacts `--out` type `bisect-*` et
  builds custom) et prend le plus récent par mtime.
- Sur `main`, la résolution était `hw_build/<board_slug>/top.fs`.
- **Risque** : `spinalml flash --board X` sans chemin explicite peut flasher le
  `top.fs` d'un autre board (ou un artefact de bisection plus récent) : mauvaise
  pinout au pire, confusion au mieux. L'affichage de la liste et le hint
  n'atténuent que le cas multi-candidats.
- **Décision mainteneur (2026-09-18)** : comportement conservé volontairement
  (confort de trouver le dernier build, pas gênant pour l'utilisateur final).
  Piste si besoin plus tard : borner à `(hw_build_root / board_slug)` d'abord.

---

## 2. Importants (à traiter ou ticketer)

### 2.1 Golden F2F (`cast_float_to_float_hw`) diverge du RTL sur NaN/sous-normal `[V]`

- `ops.py:1161-1213` : aucune branche NaN, pas de passthrough même-format.
- RTL : `cast.scala:99-103` = passthrough bit-identique si même format ;
  `Float.roundTo` canonicalise NaN (`Float.scala:621-632`, encodage
  `nanEncoding`).
- Divergence : BF16 NaN → FP8 : golden 448/`0x7E`, RTL `0x7F` ; BF16 NaN → BF16 :
  golden `0x7F80` (inf), RTL `0x7F81` ; sous-normal même-format `0x0003` :
  golden 0, RTL passthrough.
- Aucun vecteur NaN/sous-normal dans `test_cast_float.py` → trou de couverture.

### 2.2 `test_layernorm1d.py` : vérification par élément vide `[V]`

- `tests/python/test_layernorm1d.py:71-82` : la boucle calcule `exp_bits`,
  `out_bits`, puis `pass` (commentaire justificatif seulement). Le test valide
  la simulation, pas la sortie. `log_true_math_error` ne fait qu'un log.

### 2.3 Golden `RequantizeTest` dépend de la config ambiante `[V]`

- `RequantizeTest.scala:37` : `useTrunc = RoundingConfig.current == ...` ; le
  golden change avec `SPINALML_ROUNDING`. Défendable pour tourner dans les deux
  lanes, mais le test passe silencieusement dans les deux modes sans les figer.
  Préférer deux composants paramétrés par `RoundingMode`.

### 2.4 Couverture Truncate absente pour la nouvelle API `rounding` `[A]`

- `SigmoidOp`/`TanhOp` (`rounding`) : seuls RNE/défaut testés, pas de lane
  Truncate LUT. `Sequential.inputScale`/`inputZeroPoint`/`rounding` et
  `BatchNorm1D` (`shift`/`rounding`) ne sont pas exercés via LayerSpec.
- `SqrtOp`/`RsqrtOp` : tests compile-only + négatif→0, aucun golden positif
  RNE/trunc.

### 2.5 Preuve formelle du diviseur entier supprimée `[A]`

- `DivFormal.scala` : `DivFormal_I8` + `doVerify("div_i8")` supprimés alors que
  `div.scala` est réécrit (~290 lignes, chemins comb + sériel, 4 largeurs).
  Remplacés par 5 vecteurs de simulation.
- Voir aussi §4 : la revue fonctionnelle de `div.scala` n'a pas été faite.

### 2.6 `examples/Mnist/inference.py` : `encode_e4m3` en half-up `[A]`

- `inference.py:65-85` : `floor(x+0.5)` alors que le défaut HW est RNE ; le
  self-test compare HW vs NumPy, donc les égalités (runs de logits identiques
  après normalisation) peuvent donner des mismatches fantômes. Aligner sur
  `golden_models.dtypes`.

### 2.7 Self-test A/B partiellement aveugle `[A]`

- `inference.py:422-426` : `exact`/`agree` ne comparent que HW vs NumPy, pas
  RNE vs trunc ; le commentaire admet que les deux bitstreams donnent presque
  toujours la même sortie. La régression d'arrondi visée peut passer inaperçue.

---

## 3. Mineurs / hygiène

| # | Fichier(s) | Constat | Statut |
|---|---|---|---|
| 3.1 | `spinalML/test/.../ops/CastTest.scala:75-77,93` | Commentaire/nom/message « 136 » faux : l'entrée (exp 134, mant 13 = 141.0) donne RNE `exp=14 mant=1` = **144** (136 n'est même pas représentable en E4M3). L'assertion est correcte. | `[V]` |
| 3.2 | `tests/python/test_cast_float.py:66` | Même commentaire « 140.5 → 136 » faux, + « −0.0 → +0 » pour `(0,0,0)`. | `[A]` |
| 3.3 | `symbolicTest/ops/RequantizeFormal.scala:53-54,109-110` | Bornes `>= S(-128,8)` / `<= S(127,8)` tautologiques sur un payload 8 bits : ne peuvent jamais échouer (« defense in depth » illusoire). | `[V]` |
| 3.4 | `nn/LayerSpecTest.scala:40` | `check("Linear weight", hw.io.w.shape, spec.getWeightShape())` : compare l'argument à lui-même (auto-référentiel). | `[A]` |
| 3.5 | `RoundingPolicyTest.scala:39,63,65` | `StreamReadyRandomizer` non seedé + `Random.nextInt` non seedé ; attente non bornée (`waitSamplingWhere` sans timeout) → peut hangs au lieu d'échouer. | `[A]` |
| 3.6 | `tests/python/golden_models/ops.py:1090,1118,1170,1224` | Bloc de résolution `SPINALML_ROUNDING` copié 4× au lieu d'appeler `resolve_rounding` → risque de drift. | `[A]` |
| 3.7 | `tests/python/utils/tb_utils.py:15-22` | Docstring dit refléter la propriété JVM `spinalml.rounding`, mais seule l'env var est lue ; logique dupliquée. | `[A]` |
| 3.8 | `tests/python/test_cast_float.py:88` + imports | `run_cast_float_sim(..., request=None)` ignore `request` ; `pytest` importé inutilisé ; `cleanup_verilog` non importé → fixture autouse absente. | `[A]` |
| 3.9 | `ops.py:1093-1100` vs `RequantizeMath.shiftSaturate` | Cas `shift >= in_bits` sur un tie `INT_MIN` : golden 0, Scala −1. Non exercé par les tests. | `[A]` |
| 3.10 | `SigmoidTest.scala:117`, `TanhTest.scala:117` | « is still refused » faux pour I16 (accepté sur `main`) ; commentaires formels `I8 suite re-added` trompeurs (les suites I10 sont supprimées). | `[A]` |
| 3.11 | `MatmulTest.scala:242-283` | Le deadlock « dense unpadded B ne produit jamais » est figé comme contrat attendu (pas d'`assertThrows` à l'élaboration, contrairement à Concatenate/Slice/Attention). | `[A]` |
| 3.12 | `cli.py` `build` | `import os as _os` local alors qu'`os` est déjà importé en tête ; `_resolve_rounding` accepte toute valeur inconnue comme RNE sans validation/erreur. | `[V]` |

---

## 4. Non couvert par cette passe

- Revue fonctionnelle de `ops/div.scala` (réécriture ~290 lignes), du chemin
  `requantize.scala` (`shiftSaturate`), de `math_luts.scala`/`PWL.scala`
  (mapping `Truncate → Math.round`), de `sigmoid.scala`/`tanh.scala`, du câblage
  `nn/Sequential.scala`/`LayerSpec.scala`, et de `batchnorm.scala`/`layernorm.scala`.
- `scripts/diff_hw_json.py` (+518 lignes, outil de diagnostic).
- Cohérence documentaire globale (hors docs déjà touchées par la PR).

## 5. Priorisation suggérée

1. ~~**1.1 `Float.toSInt`**~~ : **corrigé et vérifié** (RTL + golden, suites
   Scala et Python vertes). Reste à committer.
2. ~~**1.2 `flash_runner`**~~ : **accepté par design** (décision mainteneur).
3. **2.5 DivFormal** : soit restaurer une preuve formelle I8 sur le nouveau
   diviseur, soit ticketer explicitement (la réécriture est le plus gros
   changement du lot).
4. 2.1/2.2/2.4 : trous de couverture à combler ou ticketer.
5. Le reste : hygiène, peut partir en issues.
