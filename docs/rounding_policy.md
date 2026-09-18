# Politique d'arrondi — RNE par défaut, troncature opt-in

> Statut : décision de design validée (2026-09-16), à implémenter dans la Wave 4
> (`SemanticsRounding`). Voir `docs/wave4_wave5_plan.md` pour l'exécution et
> `docs/wave6_ddr_scaling_plan.md` pour la suite.

## 1. Décision

**Arrondi au plus proche, ties-to-even (RNE) par défaut, avec saturation en cas
de dépassement.** Un mode **troncature** reste disponible en opt-in pour les
cibles où chaque LUT compte (choix d'élaboration, coût matériel nul en mode
troncature, chemin RNE absent du netlist).

Justification par les références :

- **NumPy** (`np.round`, doc officielle) : *« For values exactly halfway between
  rounded decimal values, NumPy rounds to the nearest even value. Thus 1.5 and
  2.5 round to 2.0, -0.5 and 0.5 round to 0.0 »* → RNE.
- **PyTorch** (`torch.round`, doc officielle) : *« This function implements the
  “round half to even” to break ties »* → RNE. C'est la primitive d'arrondi des
  conversions float → format compact.
- **FP8** (arXiv:2209.05433, NVIDIA/Arm/Intel) : *« Rounding mode (round to
  nearest even, stochastic, etc.) choice is orthogonal to the interchange format
  and left up to the implementation »* — le format n'impose rien, mais la
  saturation sur overflow est explicitement la convention retenue pour E4M3.
- **TFLite** (spec officielle) : ne mandate aucun mode d'arrondi ; les kernels
  historiques utilisent half-away. On s'aligne sur numpy/PyTorch, pas sur TFLite,
  sauf pour les *scales* des sigmoïdes quantifiées (voir Wave 5).

Le choix RNE supprime un biais systématique (−0,5 LSB en troncature, +0,5 LSB
asymétrique en half-up) et aligne la bibliothèque sur la référence
numpy/PyTorch, pour un coût matériel marginal (voir §4).

## 2. Champ d'application

| Domaine | Politique |
|---|---|
| Arithmétique FloatML (`mul`, `add`, `roundTo`, `widen`) | RNE — **déjà** le style maison (guard/sticky) |
| Cast `SInt → FloatML` (`Float.fromSInt`) | RNE (guard/sticky) — actuellement troncature (DTYPE-06) |
| Cast `FloatML → FloatML` / `FloatML → SInt` (`CastOp`) | `widen` (exact) / `roundTo` (RNE) — actuellement non branché (OPS-10) |
| Requantification entière (accus → dtype) | `shift + RNE + saturation symétrique`, un seul datapath partagé `RequantizeMath` (`RequantizeOp`, `BatchNorm1D` SInt, `AvgPool1D/2D` int — fait, Wave 5 step 3 ; saturation inatteignable sur un avgpool, gardée par partage) |
| Génération des LUT (élaboration) | RNE (`roundRNE` half-even), aligné sur `round()` Python — switch-aware (Option B, fait) ; `Truncate` = legacy `Math.round` half-up bit-exact |
| Saturation | Toujours vers la valeur max représentable (finie) ; jamais de wrap |
| `sqrt`/`rsqrt` d'une entrée négative | Politique de domaine : résultat 0 (voir OPS-02) |
| Sous-normaux | Flush-to-zero (FTZ) documenté, voir §5 |

## 3. Switch `RoundingMode` (élaboration uniquement)

```scala
sealed trait RoundingMode
case object Rne      extends RoundingMode   // défaut, aligné numpy/PyTorch
case object Truncate extends RoundingMode   // legacy / minimal LUT

object RoundingConfig {
  def current: RoundingMode =
    (env SPINALML_ROUNDING / prop spinalml.rounding) match {
      case "trunc" | "truncate" | "floor" => Truncate
      case _                              => Rne
    }
  def fromString(s: String): RoundingMode
}
```

- **Principe** : paramètre d'élaboration (jamais un signal runtime). En mode
  `Truncate`, le chemin RNE n'est pas élaboré : **0 LUT ajoutée**.
- **Précédence** : paramètre explicite de l'op > `SPINALML_ROUNDING` /
  `-Dspinalml.rounding` > défaut RNE.
- **Per-op** : `RequantizeOp(..., rounding: RoundingMode = RoundingConfig.current)`,
  idem `Float.fromSInt`, `CastOp`, `batchnorm`, `avgpool1d/2d` (chemin entier,
  spec `rounding: Option[RoundingMode] = None` plombée par `Sequential`).
  À l'élaboration : `if (rounding == Truncate) shiftLegacy else shiftRNE`.
  Le shift+RNE+saturation entier est factorisé dans `ops/requantize.scala`
  (`RequantizeMath.shiftSaturate` SInt, `shiftRound` UInt) : `RequantizeOp`,
  `BatchNorm1D` et les deux pools partagent une seule implémentation. Sur un
  avgpool la saturation est mathématiquement inatteignable (la moyenne d'un
  fenêtre reste dans la plage d'entrée et l'accumulateur fait `w + shift` bits) ;
  elle est conservée par partage, jamais par nécessité.
- **LUT/ROM (Option B, env-only)** : `MathLUTs.generateFloatMantissaROM/intEncodeFn/floatEncodeFn`,
  `PWLLUTs.generateROMs` + `UnaryPWLOp`, et les 5 feuilles `ExpOp/ReciprocalOp/SqrtOp/RsqrtOp/LogOp`
  (param `rounding` défauté, compagnons `apply` idem, sites inline sqrt/rsqrt/log dont la
  constante Q0.16 `log2ToBase`). Les composés (`DivOp`, `SigmoidOp`, `TanhOp`, `Softmax1D`,
  `LayerNorm1D`) sont **volontairement sans override** : primitifs internes sans `LayerSpec`,
  ils suivent l'env via les défauts (un futur besoin par-couche passera par une spec,
  comme Cast/Requantize/BatchNorm).
- **Per-layer** : `Requantize(shift, targetType, rounding: Option[RoundingMode] = None)`
  et `Cast(..., rounding = None)` ; `None` = config globale.
- **CLI** : `--rounding {rne|trunc}` sur `generate`/`build` (miroir de
  `--no-dsp`, `cli/spinalml_cli/cli.py:199-203`), affiché dans le bandeau.
- **Harness de test** : `tests/python/utils/tb_utils.py` lit `SPINALML_ROUNDING`
  pour sélectionner le golden (défaut RNE). Le replica Scala du moteur
  universel (`spinalML.replica.HWArithmetic.fromSInt/fromDouble/fmul/fadd`,
  `LayerReplicas.requantizeInt/castIntToFloat`) suit la même config : il élit
  RNE par défaut et retombe sur la troncature legacy en mode trunc, comme le
  RTL (`TransformHandlers` transmet en plus le `rounding` par-layer de
  `Cast`/`Requantize`).
- **Garantie legacy** : le chemin trunc reste **bit-identique** à l'existant —
  échappatoire pour les utilisateurs après le changement de sémantique par
  défaut (le défaut RNE change les sorties bit-exactes des modèles actuels,
  c'est voulu).

## 4. Coût LUT

Hypothèses : FPGA LUT6, logique fusionnée dans l'étage existant, sans pipeline
supplémentaire. Référence design W4A8 : ~31,7 k LC (cf. `full_roadmap.md`),
1 DSP ≈ 300–500 LUT équivalent.

| Décision | Δ LUT6 / élément | Δ FF | Δ ROM | Δ DSP |
|---|---|---|---|---|
| Requantize RNE (guard/sticky + incrément étroit) | 10–15 | 0 (ou 1 étage si timing) | 0 | 0 |
| `fromSInt` RNE (sticky + incrément mantisse + carry) | 12–18 / voie | 0 | 0 | 0 |
| Génération ROM RNE (constantes d'élaboration, Option B) | 0 | 0 | 0 | 0 |
| DTYPE-07 saturation E4M3 448 (params Scala compile-time) | 0 | 0 | 0 | 0 |
| OPS-02 domaine négatif (1 mux) | 1–2 / voie | 0 | 0 | 0 |
| FTZ documenté / `require` / docs / tests | 0 | 0 | 0 | 0 |

À l'échelle d'un modèle (ex. 5 sites × 8 voies) : **~0,3–1 k LUT6 = 1–3 %** ;
ROM et DSP strictement inchangés.

**Point contre-intuitif** : sur accumulateur large (I32), RNE guard/sticky est
**moins cher que half-up** (pas d'additionneur 32 bits, seulement un OR-tree +
incrément étroit). Classement : troncature (0, biaisée) < **RNE (10–15)** <
half-up (16–32) < half-away (18–36) < stochastique (30–50 + RNG).

Caveats : chiffres LUT6 (×1,5 sur LUT4) ; si l'étage de narrowing est déjà sur
le chemin critique, ajouter un registre (`largeur × lanes` FF/site) ; le coût
dominant du projet reste la table de registres MatMul.

## 5. Divergences assumées et tracées

- **Flush-to-zero (FTZ)** : IEEE 754 définit des sous-normaux (exposant 0,
  mantisse ≠ 0) sous le plus petit normal 2^(1−bias). SpinalML les traite comme
  zéro partout (`exponent == 0 ⇒ 0` dans add/mul/rsqrt, `floatEncodeFn` encode 0
  sous le min normal). Exemple FP8 E4M3 : min normal 2⁻⁶ ≈ 0,0156, le format
  définit des sous-normaux jusqu'à 2⁻⁹, que nous encodons 0. PyTorch
  `float8_e4m3fn` les représente → **divergence assumée**, non implémentée
  (chantier multi-jours, Wave 5 « optionnel »). Le résiduel LAY-04
  (diff² sous-flue alors que diff ≠ 0) est corrigé sans sous-normaux : quand
  `enc(1e-5) == 0`, LayerNorm clampe eps au plus petit normal représentable
  (E4M3 `2^-6`, E2M1 `1.0`), BF16 gardant 1e-5 (Wave 5 step 4).
- **NaN E4M3** : convention `e4m3fn` = un seul NaN (mant=111), pas d'infini,
  saturation à 448. Wave 5 (propagation seule, fait) : `Float.mul/add/gt/roundTo/widen`
  reconnaissent le slot (helpers `isNaN`/`nanEncoding`) et le propagent (signe du
  premier opérande NaN, `gt` toujours False, `max` hérite `Mux(gt,a,b)` donc le
  second opérande gagne sur NaN — asymétrie documentée) ; autres formats : exposant
  all-ones + mantisse ≠ 0 → NaN canonique `(all-ones, 1)`. Jamais d'émission
  spontanée (saturation → 448/inf inchangée). **Piège documenté** : les chemins
  LUT/PWL ne propagent pas — LUT float ≤ 8 bits décodent les bits NaN en 480.0
  (convention golden), LUT int et coefs PWL tombent à 0/saturé. Goldens Python
  inchangés (`from_float` flushe NaN→0 côté stimulus, `(15,7)` décode toujours 480).
- **TFLite** : half-away historique dans les kernels (non mandaté par la spec).
  Référence retenue uniquement pour les scales quantifiées des sigmoïdes
  (Wave 5 : `LOGISTIC scale=1/256 zp=−128`, `TANH scale=1/128 zp=0`).

## 6. Vérification

- **`RoundingPolicyTest`** (Scala) : cas de tie ±0,5 (1.5→2, 2.5→2, −0.5→0,
  0.5→0), sweep exhaustif des 256 entrées pour tout type ≤ 8 bits, dans les
  **deux modes** ; vérification que le mode trunc reste bit-exact legacy.
- **CI** : défaut RNE (toutes suites) ; lane dédiée `SPINALML_ROUNDING=trunc`
  sur les suites concernées (requantize, cast, batchnorm) → bit-exact legacy.
- **Goldens** : paramétrés par mode (une fonction, pas de duplication) ; un
  commit = un re-baseline, jamais plusieurs à la fois.
- **Mesure d'aire** : une synthèse Yosys comparative (top représentatif, RNE vs
  trunc) ; chiffres consignés ici (preuve « 0 LUT en trunc, ~1–3 % en RNE »).
- **Formels** : suites existantes en RNE par défaut, inchangées.

## 7. Backlog lié

1. **Division entière — sémantique ONNX (fait, Wave 5 steps 5A + 5B)** :
   `DivOp` int = `Div` ONNX **exact** (troncature vers zéro, I/O même dtype,
   saturation sur diviseur 0 et `INT_MIN/-1`), float inchangé. SInt/UInt ≤8 bits :
   divider restoring déroulé combinatoire (`IntDiv`, 1 cycle). >8 bits (I16/I32) :
   divider restoring série (FSM, 1 beat en vol, latence fixe `width + 2` cycles,
   backpressure par `ready`), bit-identique au chemin combinatoire. Décision
   actée : **ONNX est la référence normative** (RNE pour `QuantizeLinear` comme
   notre défaut, div exacte) ; TFLite ne sert que là où ONNX n'a rien (LUT
   sigmoïde/tanh). Reste : Sigmoid/Tanh quantifiés TFLite (step C,
   `LOGISTIC 1/256 zp=-128`, `TANH 1/128 zp=0`). Q15 LUT abandonné (ONNX ne
   définit pas de division quantifiée, TFLite n'a pas d'op runtime standard).
2. **Propagation NaN e4m3fn — fait (Wave 5 step 1)** : voir §5.
3. **Sous-normaux** : uniquement si un modèle le justifie ; FTZ reste le défaut.
4. **Rounding avgpool int — fait (Wave 5 step 3)** : RNE via le switch,
   `RequantizeMath` partagé avec `RequantizeOp` (SInt `shiftSaturate`, UInt
   `shiftRound`), specs `AvgPool1D/2D(rounding = None)` plombées par `Sequential`,
   réplica `LayerReplicas.avgPool*Int(outBits, rounding)` via `requantizeScalar`,
   goldens `avgpool1d_hw`/`avgpool2d_hw` paramétrés par mode, tests RNE + lane
   trunc.
5. **Résiduel FP8 LAY-04 — fait (Wave 5 step 4)** : eps = `enc(1e-5)` si
   représentable sinon plus petit normal (E4M3 `2^-6`, E2M1 `1.0`), réplica +
   golden alignés.
