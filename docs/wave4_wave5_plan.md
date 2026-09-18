# Plan Wave 4 (`SemanticsRounding`) + Wave 5 (post-arrondi)

> Statut : plan d'exécution validé (2026-09-16). La politique est dans
> `docs/rounding_policy.md` ; le cap DRAM est dans
> `docs/wave6_ddr_scaling_plan.md`.

## 0. État de départ

- Waves 1-3 closes : 22 fixes + 14 faux positifs annotés dans `docs/bugs/list_bug.md`.
- **Fix 24** (hors numérotation Wave 4) : mémoire des adapters en **un seul port
  d'écriture + masque byte** — BSRAM 64/46 OVER → 27/46, synthèse ~25 s,
  bitstream validé sur silicium. Règle actée : tout `Mem` ≤ 2 ports, et la
  largeur du masque pilote la largeur des symboles.
- Branche de travail : **`SemanticsRounding` depuis `main`** (après merge de
  `FixAllBugs`).
- Reste de l'audit initial = **9 entrées Wave 4** : DTYPE-06, OPS-02/03/04/05/07/10/12,
  LAY-05.

## 1. Switch d'arrondi (résumé)

Voir `docs/rounding_policy.md` §3. En bref : `RoundingMode = Rne | Truncate`,
paramètre **d'élaboration** (0 LUT en trunc), précédence op > env/prop > RNE,
`--rounding {rne|trunc}`, harness de test via `SPINALML_ROUNDING`, chemin trunc
bit-identique au legacy.

## 2. Wave 4 — 10 commits, un par étape

**Protocole** : test rouge d'abord → fix minimal → non-régression ciblée →
annotation `docs/bugs/list_bug.md` → **PAUSE** (l'utilisateur commit).

| # | Commit | Contenu | Test rouge | Goldens / re-baseline |
|---|---|---|---|---|
| 0 | **Policy + switch** | `RoundingMode`/`RoundingConfig` (calqué sur `Target.current`), helper `roundRNE` élaboration, `RequantizeOp` + spec `Requantize` avec param `rounding` (défaut RNE, trunc = legacy), `RoundingPolicyTest` (ties ±0,5, sweep 256, 2 modes), flag CLI, `tb_utils` mode, ce doc + `rounding_policy.md` | ties + sweep sur le chemin trunc (doit rester legacy) | `tests/python/test_requantize.py` (RNE devient le défaut) |
| 1 | **DTYPE-07** | Saturation E4M3 → **448** (helper `satEncoding(expBits, mantBits)` compile-time aux 8 sites RTL : `Float.mul/add/fromDouble/fromSInt/roundTo`, `math_luts`, `reciprocal`) ; NaN mant=111 laissé documenté (décodé 480) | cas de saturation E4M3 (exp/softmax extrêmes) | `golden_models/dtypes.py::from_float` |
| 2 | **OPS-07** | Deadlock MatMul sur `K % lanes != 0` quand B vient de la mémoire sans padding : trancher le contrat (padding zéro des poids dans `Sequential` **ou** gestion du K partiel dans `MatMulOp`), documenter ; OPS-09 (FP) a établi que le padding ligne est le contrat, pas l'inverse | MatMul `K=3, lanes=2` avec B non rembourré (doit bloquer avant) | — |
| 3 | **OPS-04** | `ConcatenateAxis0Op`/`SliceAxis0Op` : compteurs sur `product(shape.tail)` (battements) et non `shape.head` ; `require` cohérent sur `lanes` | tenseur 2D `lanes = 1` | — |
| 4 | **OPS-05** | `ConcatenateAxis1Op` : vérifier/forcer `lanes == shape(1)` (repack ou `require` selon le contrat retenu) | `lanes < shape(1)` | — |
| 5 | **OPS-02** | `sqrt`/`rsqrt` d'une entrée négative → 0 (mux signe), politique de domaine documentée | entrée négative FP8/BF16 | goldens ops correspondants |
| 6 | **DTYPE-06** | `Float.fromSInt` : guard/sticky + incrément mantisse + carry exposant (RNE, switch-aware) | conversion I8/I16 → FP8 exactement à mi-mantisse | `ops.py::cast_hw` (« truncated mantissa rounding ») |
| 7 | **OPS-10** | `CastOp` : brancher `FloatML → FloatML` via `widen`/`roundTo` (switch-aware), `FloatML → SInt` si le contrat le permet | cast BF16 → FP8 dans un `Sequential` (élaboration refusée avant) | tests cast |
| 8 | **LAY-05** | `BatchNorm1D` SInt : réutiliser `RequantizeOp` (**shift + rounding + saturation**), switch-aware ; changement de sémantique volontaire (wrap → sat) | vecteur de sortie dépassant la plage (wrap avant) | `ops.py::batchnorm_hw` (wrap 2's complement → saturation), tests 2 modes |
| 9 | **OPS-03/12** | `require` explicite sur `DivOp`/`SigmoidOp`/`TanhOp` pour les types entiers (message clair à l'élaboration) + entrée backlog roadmap « division Q-format + scales TFLite » | élaboration d'un div/sigmoid int (doit lever) | — |

### Détails de re-baseline

- **DTYPE-06 / OPS-10** : le golden `cast_hw` documente explicitement la
  troncature ; l'aligner sur RNE **en même temps** que le RTL (un seul commit).
- **LAY-05** : `tests/python/golden_models/ops.py` (batchnorm) reproduit le wrap
  2's complement **exprès** ; le re-baseline vers RNE+sat est le cœur du commit.
- **Switch** : chaque commit teste les deux modes quand l'op est concernée
  (RNE par défaut + lane `SPINALML_ROUNDING=trunc` bit-exact legacy).

## 3. Vérification transverse

- **CI** : défaut RNE sur toutes les suites.
- **Lane trunc** : `SPINALML_ROUNDING=trunc` sur requantize/cast/batchnorm.
- **Moteur universel** : le replica oracle (`spinalML.replica.*`) n'avait pas
  suivi les commits RNE — `HWArithmetic.fromSInt` tronquait la mantisse
  (27→26 au lieu de 28 en FP8) et `LayerReplicas.requantizeInt` faisait un
  shift sec (235>>1=117 au lieu de 118) ; les démos `UniversalMixed2DDemo` et
  `UniversalResidualDemo` échouaient donc en RNE (vertes sur `main`). Re-baseline
  switch-aware (voir `rounding_policy.md` §3) : suite universelle 10/10 en RNE,
  lane trunc revalidée sur Residual/Mixed2D.
- **Option B transverse (générateurs LUT/ROM switch-aware)** : `roundRNE` branché
  aux 7 sites d'élaboration (`generateFloatMantissaROM`, `intEncodeFn`,
  `floatEncodeFn`, inline sqrt/rsqrt/log dont `log2ToBase`), param `rounding`
  défauté sur `generateROMs`/`UnaryPWLOp` et les 5 feuilles
  (Exp/Reciprocal/Sqrt/Rsqrt/Log), composés en env-only volontaire. Goldens
  Python threadés (`pwl_*`, `SIntML/FloatML.from_float`, idiome
  `rounding=None→env`) ; suites exp/reciprocal/sqrt/rsqrt/log + formels +
  universel softmax/activations en RNE et lane trunc ; preuve MNIST inchangé
  (`md5sum` builds RNE/trunc identiques — aucun de ces générateurs n'alimente
  son datapath).
- **Garde générique LAY-02 (conformité `LayerSpec` ↔ IO HW, step 2)** :
  `LayerSpecTest` « LayerSpec metadata matches hardware ports (LAY-02) »
  (`LayerSpecConformanceComp`) compare à l'élaboration les ports d'un composant
  par famille de spec (Linear, Conv1D/2D, BatchNorm1D, LayerNorm1D,
  ClassicalAttention) aux `getWeightShape()/getBiasShape()/getOutShape()` —
  relations non tautologiques (Conv reconstruit depuis K/inC/outC, attention
  = somme des 4 ports `[embedDim, embedDim]` sans biais). Second test
  « weightless LayerSpecs declare no weight/bias region (LAY-02) » verrouille
  le trigger DMA `Sequential.scala:204` (`head > 0`). Test-only, aucun
  changement RTL ; sensibilité prouvée en rouge par injection temporaire de
  `Seq(4,1)` (`IllegalArgumentException: LAY-02 conformance …`).
  Non-régression : `test-all -k "LayerSpecTest|SequentialTest|AcceleratorTest"`
  4/4 ✅.
- **Aire** : une synthèse Yosys comparative (top représentatif, RNE vs trunc) ;
  chiffres consignés dans `docs/rounding_policy.md`.
- **Formels** : suites existantes en RNE, inchangées ; ajouter au besoin un
  invariant sur la saturation E4M3 (DTYPE-07).
- **Garde-fou synthèse** (issu de Fix 24) : smoke test CI sur le build Mnist/tang
  — échec si `using FF mapping`, BSRAM > 46, ou Fmax < cible.

## 4. Wave 5 (post-arrondi, hors DDR)

Ordre proposé :

1. **Division Q-format + sigmoid/tanh quantifiés** (~3–4 j, PR dédiée) :
   `fracBits`/scales, d'abord I8 en Q15 (LUT réciproque + RNE + saturation),
   diviseur itératif si > 8 bits ; sémantique de probabilité quantifiée alignée
   TFLite (`LOGISTIC scale=1/256 zp=−128`, `TANH scale=1/128 zp=0`) ; change
   `LayerSpec` + goldens.
2. **NaN `e4m3fn` complet** (mant=111) : reconnaissance/propagation dans
   mul/add/gt/roundTo (~1 j + goldens), pendant de DTYPE-07.
3. **Test de conformité générique `LayerSpec` ↔ IO HW** (classe LAY-02) : pour
   chaque `LayerSpec`, vérifier forme/type des poids/biais contre les ports
   matériels instanciés (~1 j).
4. **Résiduel FP8 LAY-04** : sous-flux `diff²` avec `diff ≠ 0` ; eps min-normal
   représentable + décision golden.
5. **Rounding avgpool int** : division shiftée arrondie (RNE via le switch) si
   la sémantique entière compte.
6. **Sous-normaux** (optionnel lourd, multi-jours) : uniquement si un modèle le
   justifie ; FTZ reste le défaut.
7. **Petites dettes** : références de docs périmées, généricité, messages
   d'erreur d'élaboration.

## 5. Risques / garde-fous

- Le switch **ne doit jamais** être un signal runtime (coût LUT partout) —
  vérifié par le synth comparatif.
- Goldens paramétrés par mode, pas dupliqués ; un commit re-baseline **une**
  sémantique à la fois.
- `RequantizeOp` reste `SInt → SInt` ; BatchNorm réutilise, ne duplique pas.
- Le défaut RNE **change les sorties bit-exactes** des modèles actuels (voulu) ;
  `--rounding trunc` restaure l'ancien narrowing (pas les fixes structurels
  1–5).
- Ne pas fusionner des fixes d'hygiène non liés dans cette branche sémantique
  (risque de mélange de re-baselines).
