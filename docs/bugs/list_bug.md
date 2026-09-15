# Liste Complète des Bugs Potentiels, Anomalies et Incohérences Post-Refactoring

Ce document recense l'ensemble des bugs potentiels, comportements anormaux, régressions possibles et incohérences architecturales identifiés lors de la revue minutieuse de tous les fichiers `.scala` du projet **spinalML**.

---

## Sommaire
- [1. Types de Données & Arithmétique Flottante (`dtypes`, `utils/Float.scala`, `utils/math_luts.scala`)](#1-types-de-données--arithmétique-flottante)
- [2. Opérations Fondamentales & Algèbre (`ops`)](#2-opérations-fondamentales--algèbre)
- [3. Mémoire, Buffers, DMA & Adaptateurs (`memory`)](#3-mémoire-buffers-dma--adaptateurs)
- [4. Fonctions d'Activation & Poolings (`activations`, `poolings`)](#4-fonctions-dactivation--poolings)
- [5. Couches de Haut Niveau & Attention (`layers`, `attention`)](#5-couches-de-haut-niveau--attention)
- [6. Réseau Séquentiel, DAG & Accélérateur (`nn`)](#6-réseau-séquentiel-dag--accélérateur)
- [7. Entrées / Sorties, Pont Série & SoC (`io`)](#7-entrées--sorties-pont-série--soc)
- [8. Utilitaires & DSL (`utils`, `nn/DagDSL.scala`)](#8-utilitaires--dsl)

---

## 1. Types de Données & Arithmétique Flottante

### BUG-DTYPE-01 : Crash d'élaboration sur tranche négative dans `Float.roundTo`
- **Statut** : faux positif (vérifié le 2026-09-15) — aucune exception : `mantExt(-1 downto 0)` produit un slice de largeur 0 via `RangePimper` (`SpinalHDL core.scala:507-519`), et `(0 bits) =/= 0` se constant-fold en `False`, soit exactement le `sticky` RNE attendu pour `drop == 1`. Test d'élaboration ajouté : `spinalML/test/src/spinalML/utils/FloatTest.scala` (« roundTo drops a single mantissa bit »).
- **Fichier** : `spinalML/src/spinalML/utils/Float.scala` (ligne 397)
- **Code concerné** :
  ```scala
  val drop = a.mantBits - outMantBits
  ...
  val sticky = mantExt(drop - 2 downto 0) =/= 0
  ```
- **Description** :
  Lorsque `drop == 1` (par exemple lors de la conversion de `FP8_E4M3` vers `FP8_E5M2`, ou `FloatML(exp, 4)` vers `FloatML(exp, 3)`), `drop - 2` s'évalue à `-1`. L'expression devient `mantExt(-1 downto 0)`.
- **Conséquence** :
  SpinalHDL lève une exception fatale à la génération (`high < low`) et fait planter la compilation du matériel.
- **Correction recommandée** :
  Remplacer par :
  ```scala
  val sticky = if (drop >= 2) (mantExt(drop - 2 downto 0) =/= 0) else False
  ```

---

### BUG-DTYPE-02 : Crash d'élaboration sur tranche négative dans `Float.mul`
- **Statut** : faux positif (vérifié le 2026-09-15) — même mécanisme que DTYPE-01 : `mantProd(-1 downto 0)` donne un slice de largeur 0 via `RangePimper` (`SpinalHDL core.scala:507-519`), donc le `stickyM` de la branche non-overflow est constant `False` (Vérifié dans le Verilog généré : `(overflow ? (prod[0] != 0) : 1'b0)`), soit le RNE correct pour un seul bit de mantisse. Test d'élaboration ajouté : `spinalML/test/src/spinalML/utils/FloatTest.scala` (« mul on a single-mantissa-bit format (FP4_E2M1) »).
- **Fichier** : `spinalML/src/spinalML/utils/Float.scala` (ligne 51)
- **Code concerné** :
  ```scala
  val stickyM = Mux(overflow,
    (mantProd(mantBits - 1 downto 0) =/= 0),
    (mantProd(mantBits - 2 downto 0) =/= 0)
  )
  ```
- **Description** :
  Pour les formats ultra-compacts possédant 1 seul bit de mantisse (tel que `FP4_E2M1`), `mantBits == 1`. L'expression `mantBits - 2` vaut `-1`.
- **Conséquence** :
  Crash d'élaboration SpinalHDL immédiat sur `mantProd(-1 downto 0)` dès qu'une multiplication est instanciée sur un format à 1 bit de mantisse.
- **Correction recommandée** :
  Ajouter une condition scalaire :
  ```scala
  val stickyM = if (mantBits >= 2) {
    Mux(overflow, mantProd(mantBits - 1 downto 0) =/= 0, mantProd(mantBits - 2 downto 0) =/= 0)
  } else {
    Mux(overflow, mantProd(0 downto 0) =/= 0, False)
  }
  ```

---

### BUG-DTYPE-03 : Débordement d'exposant signant la perte de valeur (conversion Float étroit -> large)
- **Statut** : corrigé — bug réel, mais le mode de défaillance observé est une **erreur d'élaboration** `OUT OF RANGE CONSTANT` (la comparaison `expSInt >= (1 << outExpBits) - 1` ne tient pas dans la largeur `a.expBits + 4` dès que `outExpBits >= a.expBits + 4`, ex. FP4/FP8 -> FP32), pas l'underflow silencieux décrit initialement. Fix : `expSIntWidth = (a.expBits max outExpBits) + 4` appliqué aux deux branches (drop et widening). Tests : `spinalML/test/src/spinalML/utils/FloatTest.scala` (« roundTo widens the exponent without overflow (FP4 4.0 -> FP32 4.0) ») et non-régression `tests/python/test_softmax.py` (7 passed).
- **Fichier** : `spinalML/src/spinalML/utils/Float.scala` (lignes 403-405 et 421)
- **Code concerné** :
  ```scala
  val expSInt = a.exponent.intoSInt.resize(a.expBits + 4 bits) +
    biasDelta +
    mantOv.asUInt.intoSInt.resized
  ```
- **Description** :
  Lors de la conversion d'un petit format vers un grand format (ex. `FP4` avec `expBits = 2` ou `FP8` avec `expBits = 4`, bias = 7 vers `FP32` où `outExpBits = 8`, bias = 127), `biasDelta = 127 - 7 = 120`.
  Le redimensionnement de `a.exponent` utilise la taille d'entrée : `a.expBits + 4 = 8 bits`.
  Dans un `SInt(8 bits)`, l'intervalle représentable est `[-128, +127]`.
  Dès que `a.exponent >= 8`, la somme `a.exponent + biasDelta >= 128`, ce qui déborde la représentation signée 8 bits et devient un nombre négatif (ex. $+128 \to -128$).
- **Conséquence** :
  La condition `elsewhen(expSInt <= 0)` s'active à tort ! Un nombre flottant grand et parfaitement valide est transformé en `0.0` (underflow erroné).
- **Correction recommandée** :
  Dimensionner `expSInt` en fonction du maximum des deux exposants :
  ```scala
  val expSIntWidth = (a.expBits max outExpBits) + 4
  val expSInt = a.exponent.intoSInt.resize(expSIntWidth bits) + biasDelta + ...
  ```

---

### BUG-DTYPE-04 : Débordement dans `MathLUTs.floatEncodeFn` transformant l'Infini en Zéro
- **Statut** : corrigé — `+Inf`/`-Inf` encodés canoniquement (exposant all-ones, mantisse 0, signe préservé) et `NaN` -> 0 (même convention que le golden Python `dtypes.from_float`), au lieu du débordement `Double.toInt` qui produisait 0 (`+Inf`), `0x80` (`-Inf`) ou une mantisse poubelle (`NaN`). Tests : `spinalML/test/src/spinalML/utils/MathLUTsTest.scala` (3 cas) ; non-régression : `tests/python/test_exp.py --debug-math` (4 passed, métriques de précision inchangées).
- **Fichier** : `spinalML/src/spinalML/utils/math_luts.scala` (lignes 74-88)
- **Code concerné** :
  ```scala
  var exp = Math.floor(Math.log(absY) / Math.log(2.0)).toInt
  ...
  var expEnc = exp + bias
  ...
  if (expEnc >= (1 << expBits)) { // Overflow (Sat)
    ...
  } else if (expEnc <= 0) { // Underflow
    expEnc = 0
    mantEnc = 0
  }
  ```
- **Description** :
  Lorsque `absY` est infini (`Double.PositiveInfinity`) ou dépasse la capacité des entiers, `Math.log(absY)` vaut `Infinity`. Le cast Scala `.toInt` sur `Infinity` produit `Int.MaxValue` (`2147483647`).
  L'addition `expEnc = exp + bias` déborde l'entier signé 32 bits en valeur fortement négative (`-2147483642`).
- **Conséquence** :
  La condition `expEnc <= 0` s'évalue à `true` : l'infini est encodé sous forme de zéro absolu (`0.0`) au lieu de saturer à l'infini matériel !
- **Correction recommandée** :
  Gérer explicitement `if (y.isInfinity)` avant les calculs logarithmiques :
  ```scala
  if (y.isInfinity) {
    val sign = if (y < 0) 1 else 0
    return BigInt((sign << (expBits + mantBits)) | (((1 << expBits) - 1) << mantBits))
  }
  ```

---

### BUG-DTYPE-05 : Incohérence d'encodage de l'Infini entre `Float.scala` et `math_luts.scala`
- **Statut** : corrigé — la saturation sur débordement fini encode désormais l'infini canonique (exposant all-ones, mantisse 0), cohérent avec `Float.scala` (ligne 76) et le golden `dtypes.from_float`, au lieu de la mantisse all-ones qui correspond à un code NaN. Tests : `spinalML/test/src/spinalML/utils/MathLUTsTest.scala` (« floatEncodeFn saturates finite overflow to canonical infinity (mant 0) ») ; non-régression `tests/python/test_exp.py --debug-math` (4 passed, métriques inchangées).
- **Fichier** : `spinalML/src/spinalML/utils/math_luts.scala` (ligne 87) vs `Float.scala` (ligne 76)
- **Description** :
  Dans `Float.scala`, la saturation vers l'infini configure `exponent = (1 << expBits) - 1` et `mantissa = 0` (conformité IEEE 754).
  Dans `math_luts.scala`, la saturation configure `mantEnc = (1 << mantBits) - 1`. Selon la norme IEEE, un exposant maximal avec une mantisse non nulle correspond à un `NaN` et non à `Infinity`.
- **Conséquence** :
  Incohérence des représentations entre le calcul RTL matériel et les tables générées par LUT.

---

### BUG-DTYPE-06 : Tronquage sans arrondi dans `Float.fromSInt`
- **Fichier** : `spinalML/src/spinalML/utils/Float.scala` (ligne 315)
- **Code concerné** :
  ```scala
  val mantissa = paddedVal(W_padded - 2 downto W_padded - 1 - mantBits)
  ```
- **Description** :
  La conversion entière vers flottante extrait brutalement les bits de mantisse par troncature (`floor`) sans calculer de bit d'arrondi (`guard`, `round`, `sticky`), contrairement aux conversions IEEE 754 et au modèle doré Python.
- **Conséquence** :
  Introduction d'un biais systématique négatif lors de la déquantification ou de la conversion d'accumulateurs d'entiers vers le domaine flottant.

---

## 2. Opérations Fondamentales & Algèbre

### BUG-OPS-01 : Inversion de signe dramatique via `.intoSInt` dans `ops/log.scala`
- **Statut** : corrigé (commit `ec54afe`) — test de couverture : `tests/python/test_log.py::test_log_bf16` et `::test_log_bf16_base10` (valeurs x < 1).
- **Fichier** : `spinalML/src/spinalML/ops/log.scala` (ligne 79)
- **Code concerné** :
  ```scala
  val logFixed = (((expTrueSInt << 8).asUInt) | fracBits.resize(log2Up(mantBits + 1) + 8)).intoSInt
  ```
- **Description** :
  Dans SpinalHDL, invoquer `.intoSInt` sur un signal `UInt` ajoute un bit de signe `'0'` en tête au lieu de préserver l'interprétation en complément à deux.
  Pour toute entrée $x \in ]0, 1[$, $\log_2(x) < 0$. `expTrueSInt` est donc strictement négatif et son bit de poids fort est à `1`. En le convertissant en `UInt` par `.asUInt` puis en `SInt` par `.intoSInt`, SpinalHDL ajoute un 0 en MSB, transformant le nombre négatif en un entier positif gigantesque ($\sim +65400$).
- **Conséquence** :
  Tous les logarithmes pour des valeurs $< 1$ (fréquents en sortie de softmax ou en probabilités) sont corrompus en valeurs immenses positives au lieu de valeurs négatives.
- **Correction recommandée** :
  Remplacer impérativement `.intoSInt` par `.asSInt`.

---

### BUG-OPS-02 : Préservation aberrante du signe négatif dans `ops/sqrt.scala` et `ops/rsqrt.scala`
- **Fichier** : `spinalML/src/spinalML/ops/sqrt.scala` (ligne 55) & `spinalML/src/spinalML/ops/rsqrt.scala` (ligne 53)
- **Code concerné** :
  ```scala
  outX.sign := RegNextWhen(x.sign, io.a.stream.ready)
  ```
- **Description** :
  Le bloc calcule la racine carrée ou la racine inverse de la magnitude, puis réassigne directement le signe d'entrée `x.sign`.
- **Conséquence** :
  Si une valeur négative entre dans le bloc (par exemple `-4.0`), le circuit produit $\sqrt{-4.0} = -2.0$ et $1/\sqrt{-4.0} = -0.5$ au lieu de lever un signal d'erreur, saturer à zéro ou produire `NaN`.

---

### BUG-OPS-03 : Division et activation Sigmoid/Tanh nulles sur entiers `SInt`
- **Fichier** : `spinalML/src/spinalML/ops/div.scala`, `spinalML/src/spinalML/activations/sigmoid.scala`, `spinalML/src/spinalML/activations/tanh.scala`
- **Description** :
  `DivOp` est implémenté via `ReciprocalOp` suivi d'une multiplication. Pour les types entiers bruts non normalisés (`SInt`), `ReciprocalOp` calcule `round(1.0 / x)`. Pour tout entier $x \ge 2$, `1.0 / x < 0.5`, ce qui est arrondi à `0`.
- **Conséquence** :
  La division entière $A / B$ produit systématiquement $0$ pour tout dénominateur $B > 1$. De même, `SigmoidOp` ($1 / (1 + e^{-x})$) s'écrase à zéro pour la quasi-totalité des valeurs entières car il manque un facteur d'échelle en virgule fixe (format Qm.n).

---

### BUG-OPS-04 : Tronquage de dimensions dans `ConcatenateAxis0Op` et `SliceAxis0Op`
- **Fichier** : `spinalML/src/spinalML/ops/concatenate.scala` (lignes 21-23) & `spinalML/src/spinalML/ops/slice.scala` (lignes 20-22)
- **Code concerné** :
  ```scala
  val L_in = shape.head
  ...
  val counter = Counter(L_in)
  ```
- **Description** :
  Pour un tenseur 2D ou 3D (par exemple de forme $[H, W]$ avec $lanes = 1$), le flux transporte $H \times W$ battements. Or, le composant initialise son compteur à `shape.head` ($H$).
- **Conséquence** :
  Après seulement $H$ battements (une infime fraction du tenseur si $W > 1$), le compteur boucle ou bascule prématurément sur le tenseur suivant. Le composant n'est fonctionnel que si chaque battement transporte une tranche complète ($lanes = \prod shape.tail$).

---

### BUG-OPS-05 : Entrelacement spatial erroné dans `ConcatenateAxis1Op`
- **Fichier** : `spinalML/src/spinalML/ops/concatenate.scala` (lignes 63-70)
- **Description** :
  `ConcatenateAxis1Op` concatène les vecteurs de voies battement par battement sans vérifier que `lanes == shape(1)`.
- **Conséquence** :
  Si `lanes < shape(1)`, les éléments de différentes lignes des tenseurs A et B sont entrelacés dans le temps au lieu d'être accolés colonne par colonne.

---

### BUG-OPS-06 : Perte du parallélisme de voies dans `ops/transpose.scala`
- **Statut** : corrigé — `finalLanes = if (outLanes > 0) outLanes else a.lanes` (ligne 108) ; le repack interne vers 1 voie reste un détail d'implémentation, la sortie par défaut retrouve les lanes d'entrée. Tests : `TransposeTest.scala` (« transpose preserves the input lanes by default », top lanes = 4, rouge avant : `transpose dropped lanes: 1 != 4`) ; non-régression Scala `TransposeTest` 6/6, Python `test_transpose.py::test_transpose_4x4_i8` et `test_classicalattention.py::test_pytest_attention_bf16` (seul call site).
- **Fichier** : `spinalML/src/spinalML/ops/transpose.scala` (ligne 108)
- **Code concerné** :
  ```scala
  val in = repack(a, 1)
  ...
  val finalLanes = if (outLanes > 0) outLanes else in.lanes
  ```
- **Description** :
  Comme le tenseur `in` est forcé à 1 voie (`repack(a, 1)` à la ligne 104), `in.lanes` vaut obligatoirement `1`.
- **Conséquence** :
  Tout appel à `transpose(tensor)` retourne un tenseur à 1 voie, régressant silencieusement les performances multi-voies sauf si l'utilisateur spécifie explicitement `outLanes`.

---

### BUG-OPS-07 : Blocage (`Deadlock`) dans `MatMulOp` sur matrices non multiples de `lanes`
- **Fichier** : `spinalML/src/spinalML/ops/matmul.scala`
- **Description** :
  Le buffer B est dimensionné à `paddedK * N` et attend `chunksK * N` battements. Si le tenseur de poids B provient de la mémoire externe sans avoir été préalablement rembourré (padding par colonne), il ne délivre que $K \times N$ éléments.
- **Conséquence** :
  Le buffer B ne reçoit jamais son quota de battements ; l'état `stateTileB` ne se termine jamais et l'opération bloque indéfiniment.

---

### BUG-OPS-08 : Corruption des canaux dans `CumSumOp` lorsque la dimension $C$ n'est pas multiple de `lanes`
- **Statut** : faux positif (vérifié le 2026-09-15) — le contrat de streaming du projet rembourre **chaque ligne** à `ceil(C/lanes)` battements (même contrat que `MatmulDynPad`, `test_matmul.py`), et `CumSumOp` compte précisément `chunks` battements par ligne avec un `Delay(chunks)` : l'alignement de voie est donc préservé pour `C % lanes != 0`. Test ajouté : `CumsumTest.scala` (« CumSum with C=5, lanes=4 on per-row padded beats », valeurs par ligne vérifiées, passe). Un `require(C % lanes == 0)` casserait ce contrat légitime.
- **Fichier** : `spinalML/src/spinalML/ops/cumsum.scala` (lignes 21, 47-58)
- **Code concerné** :
  ```scala
  val chunks = (C + lanes - 1) / lanes
  ...
  val prevValDelayed = Delay(sumResult, cycleCount = chunks, when = fire, init = sumResult.getZero)
  ```
- **Description** :
  `CumSumOp` applique un délai fixe de `chunks` cycles pour retrouver les valeurs cumulées de la ligne précédente. Si la dimension intérieure $C$ n'est pas un multiple de `lanes` (ex: $C=5$ avec `lanes=4`), les éléments de la ligne suivante sont concaténés dans le flux continu dès le 2ᵉ battement et arrivent avec un décalage d'indices de voies.
- **Conséquence** :
  Les éléments d'une colonne sont additionnés avec des colonnes différentes de la ligne précédente. Le résultat mathématique de la somme cumulée est totalement faussé.
- **Correction recommandée** :
  Imposer explicitement `require(C % lanes == 0, s"CumSumOp requires C ($C) to be a multiple of lanes ($lanes)")` ou implémenter un aligneur de décalage de voies (barrel shifter).

---

### BUG-OPS-09 : Désynchronisation inter-lignes dans `MatMulOp` (mode séquentiel) lorsque $K \pmod{lanes} \ne 0$
- **Statut** : faux positif (vérifié le 2026-09-15) — le rembourrage par ligne est le contrat d'entrée : le driver de `test_matmul.py` zéro-remplit chaque ligne (`k_idx >= K -> 0.0`) et le test `test_matmul_dyn_pad_*` (K=3, lanes=2) passe déjà ; le RTL masque explicitement les lanes de padding (`matmul.scala:248`). Ajouter `require(K % lanes == 0)` casserait ce test valide.
- **Fichier** : `spinalML/src/spinalML/ops/matmul.scala` (lignes 232-249, 321-328)
- **Code concerné** :
  ```scala
  memA.write(loadACounter.value, io.a.stream.payload)
  loadACounter.increment()
  when(loadACounter.willOverflowIfInc) { goto(stateComputeN) }
  ```
- **Description** :
  Pour chaque ligne de la matrice $A$, `loadACounter` lit `chunksK` battements depuis `io.a.stream`. Si $K$ n'est pas un multiple de `lanes`, le flux plat continu commence la ligne 1 à l'intérieur du dernier battement de la ligne 0. Or, `memA` absorbe le battement entier pour la ligne 0, et commence la lecture de la ligne 1 sur le battement suivant.
- **Conséquence** :
  Toutes les lignes de $A$ après la ligne 0 sont lues avec un décalage spatial progressif, corrompant l'ensemble des multiplications matricielles sauf si $K$ est divisible par `lanes` ou que l'amont rembourre chaque ligne.
- **Correction recommandée** :
  Ajouter un `require(K % lanes == 0)` ou un module de réalignement spatial par ligne.

---

### BUG-OPS-10 : Rejet brutal des conversions `FloatML -> FloatML` et `FloatML -> SInt` dans `CastOp`
- **Fichier** : `spinalML/src/spinalML/ops/cast.scala` (lignes 88-94)
- **Code concerné** :
  ```scala
  case (valIn: SInt, valOut: FloatML) => ...
  case (valIn: SInt, valOut: SInt) => ...
  case _ => throw new Exception("Type de cast non supporté (SInt -> FloatML et SInt -> SInt sont gérés)")
  ```
- **Description** :
  Le composant matériel `CastOp` ne gère que les entrées `SInt`. Si un utilisateur essaie de convertir un tenseur flottant vers un autre format flottant (par exemple `BF16` vers `FP8`, ou `FP8_E4M3` vers `FP8_E5M2`) ou vers un entier, l'élaboration lève une exception fatale.
- **Conséquence** :
  Bien que les primitives de conversion `Float.roundTo` et `Float.widen` existent dans `utils/Float.scala`, elles ne sont pas branchées dans l'opérateur de haut niveau `Cast` / `CastOp`, empêchant tout changement de précision flottante dans un graphe `Sequential`.
- **Correction recommandée** :
  Intégrer les cas `case (valIn: FloatML, valOut: FloatML)` en appelant `Float.roundTo` ou `Float.widen`.

---

### BUG-OPS-11 : Interprétation signée erronée de l'exposant (`intoSInt`) provoquant un faux débordement dans `ReciprocalOp`
- **Statut** : faux positif (vérifié le 2026-09-15) — dans SpinalHDL 1.15.0, `UInt.intoSInt = this.expand.asSInt` et `expand` **préfixe un 0** (`core/UInt.scala:397/399`) : l'exposant biaisé est donc zéro-étendu, ce qui est le comportement correct. Les tests BF16 `1/2.0`, `1/50.0` passent. Ne pas « corriger » en `asSInt` (ce serait le bug). Preuve du piège inverse : BUG-OPS-01, où `intoSInt` est appliqué à une valeur déjà en complément à 2.
- **Fichier** : `spinalML/src/spinalML/ops/reciprocal.scala` (lignes 66-74)
- **Code concerné** :
  ```scala
  val expSInt = x.exponent.intoSInt
  val shift = Mux(mantIsZero, S(0, expBits+2 bits), S(1, expBits+2 bits))
  val newExpSInt = S(2 * bias, expBits+2 bits) - expSInt - shift
  ...
  val expOverflow = RegNextWhen(newExpSInt >= ((1 << expBits) - 1), io.a.stream.ready)
  ```
- **Description** :
  `x.exponent` est un `UInt(expBits bits)` représentant un exposant non signé avec biais. L'appel `x.exponent.intoSInt` convertit directement les bits bruts en `SInt(expBits bits)` de même largeur en complément à 2.
  Dès que le bit de poids fort de l'exposant vaut 1 (c'est-à-dire `exponent >= (1 << (expBits - 1))`, ce qui est le cas pour tout nombre réel $\ge 2.0$ avec le biais standard IEEE), `expSInt` est considéré comme un nombre négatif.
  Par exemple en BF16 (`bias = 127`), pour $x = 2.0$, l'exposant brut vaut $128$ (`10000000`b), qui est transformé par `intoSInt` en $-128$. L'évaluation `2 * bias - expSInt - shift` devient $254 - (-128) - 0 = 382$, ce qui active immédiatement `expOverflow` ($\ge 255$).
- **Conséquence** :
  L'inverse $1/x$ de n'importe quel nombre flottant $\ge 2.0$ produit l'infini (`Inf`) au lieu d'une fraction positive ($0.5$, etc.).
- **Correction recommandée** :
  Zéro-étendre l'exposant d'un bit avant de le convertir en entier signé :
  `val expSInt = (U"0" @@ x.exponent).asSInt.resize(expBits + 2)`.

---

### BUG-OPS-12 : Division entière (`DivOp`) mathématiquement nulle pour les types `SInt` / `UInt`
- **Fichier** : `spinalML/src/spinalML/ops/div.scala` (lignes 21-23, 43-47)
- **Code concerné** :
  ```scala
  val invBComp = ReciprocalOp(dataType, shape, lanes)
  ...
  case (va: SInt, vInvB: SInt) =>
    outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
  case (va: UInt, vInvB: UInt) =>
    outPayload(i).assignFrom((va * vInvB).resized.asInstanceOf[T])
  ```
- **Description** :
  Pour réaliser la division $a / b$, `DivOp` calcule $1/b$ via `ReciprocalOp` puis multiplie $a \times (1/b)$.
  Cependant, pour les types entiers `SInt` et `UInt`, `ReciprocalOp` encode $1/b$ dans le même type sans format à virgule fixe (pas de bits fractionnaires $Q_{m.n}$).
  L'arrondi entier `Math.round(1.0 / b)` donne :
  - $0$ pour tout diviseur $|b| \ge 3$ ($1/3 \approx 0.33 \to 0$, $1/4 \approx 0.25 \to 0$, etc.).
  - $1$ pour $b = 2$ ($1/2 = 0.5 \to 1$).
  - $1$ pour $b = 1$.
- **Conséquence** :
  Pour tout entier $b \ge 3$, la division $a / b$ retourne systématiquement $0$ quel que soit $a$ (ex: $100 / 5 = 0$).
  Pour $b = 2$, la division retourne $a \times 1 = a$ (ex: $10 / 2 = 10$).
  L'opérateur de division est totalement inutilisable et trompeur pour les types entiers.
- **Correction recommandée** :
  Soit refuser explicitement les types `SInt`/`UInt` à l'élaboration avec un message d'erreur clair indiquant que la division LUT n'est supportée que pour `FloatML`, soit implémenter un diviseur séquentiel ou pipeliné non-restoring/radix-2 pour les entiers.

---

## 3. Mémoire, Buffers, DMA & Adaptateurs

### BUG-MEM-01 : Décalage d'unités (battements vs éléments) dans le rognage spatial de `DMAReader2D`
- **Statut** : corrigé (2026-09-15) — `rowSkip`/`rowKeepEnd` sont désormais latés **en battements de sortie** (`headSkipElems / outLanes` et `(headSkipElems + rowWidth) / outLanes - 1`, division exacte sous le contrat « lignes alignées au groupe ») au lieu d'éléments comparés à `elemCnt` (battements). Test rouge : `DMAReader2DTest` « DMAReader2D outLanes=4 with unaligned row starts » (base 0x1004, stride 12, I8, outLanes=4 → 16/32 éléments avant fix, timeout ; 32/32 après). Formel renforcé : `DMAReader2DFormal` exprime `rowSkip`/`rowKeepEnd`/comptage en battements et une variante `Lanes2Aligned` (base/stride 2 octets alignés, skip non nul couvert) a été ajoutée ; `test-all-formal -k DMAReader2DFormal` passe. Non-régression : `DMAReader2DTest` 3/3, `test_dma_reader2d.py`, `SequentialTest`.
- **Fichier** : `spinalML/src/spinalML/memory/DMAReader2D.scala` (lignes 152-156)
- **Code concerné** :
  ```scala
  val suppress = (elemCnt < rowSkip) || (elemCnt > rowKeepEnd)
  ```
- **Description** :
  `elemCnt` compte les **battements** stream sortants, tandis que `rowSkip` et `rowKeepEnd` sont exprimés en **éléments** (pixels).
- **Conséquence** :
  Dès que `outLanes > 1`, le bloc ignore `rowSkip` battements entiers, soit `rowSkip * outLanes` éléments (supprimant beaucoup trop d'éléments). Le rognage de droite échoue également à éliminer les éléments superflus de fin de ligne.

---

### BUG-MEM-02 : Violation de protocole Stream sur `io.cmd` dans `DMAReader2D`
- **Statut** : faux positif (vérifié le 2026-09-15) — `io.cmd` est un esclave Stream : différer `ready` jusqu'à la fin de la transaction est légal, le maître devant maintenir `valid`/payload jusqu'au handshake (contrat explicitement vérifié par `DMAReader2DFormal` : `assume(cmd.valid && payload === past(payload))` quand `ready` est bas). Les bancs Scala (`waitSamplingWhere(cmd.ready)`) et cocotb (valid maintenu jusqu'à ready) sont conformes ; le scénario « maître attendant un ack immédiat » suppose un maître hors protocole.
- **Fichier** : `spinalML/src/spinalML/memory/DMAReader2D.scala` (lignes 95-103)
- **Description** :
  Dans l'état `stateIdle`, `when(io.cmd.valid)` démarre immédiatement le traitement multi-lignes mais n'asserte pas `io.cmd.ready`. Le signal `ready` n'est levé que dans l'état final `stateDrain`.
- **Conséquence** :
  Violation de la sémantique de transaction `Stream` de SpinalHDL : un maître synchrone attendant l'acquittement immédiat de sa commande reste bloqué ou désynchronisé.

---

### BUG-MEM-03 : Écriture mémoire non alignée et écrasement par strobes pleins dans `DMAWriter`
- **Statut** : corrigé (2026-09-15) — `io.axiMaster.w.strb` n'est plus `setAll()` : le dernier beat du transfert (`remaining === 0 && burstRemain === 1`, le transfert étant contigu le beat partiel est toujours le dernier) reçoit le masque des `(totalElements - (totalAxiBeats-1)*axiLanes) * (elemWidth/8)` octets valides, les beats pleins gardent tout-1. Test rouge : `tests/python/test_dma_writer.py::test_dma_writer_partial_last_beat` (3 SInt16 = 6 octets sur bus 64 b, sentinelle 0xBEEF dans les octets 6-7 : avant fix `strb=0xFF` et sentinelle écrasée ; après fix `strb=0x3F` et sentinelle intacte). Le banc `AxiWriteSlave` applique désormais `w.strb` octet par octet (il ignorait le strobe, d'où l'angle mort). Non-régression : `test_dma_writer.py` 4/4, `test-all -k DMAWriterTest`, `test-all-formal -k DMAWriterFormal`, `test-all-python -k accelerator` (write-back).
- **Fichier** : `spinalML/src/spinalML/memory/DMAWriter.scala` (ligne 96)
- **Code concerné** :
  ```scala
  io.axiMaster.w.strb.setAll()
  ```
- **Description** :
  Le masque d'écriture AXI `wstrb` est fixé à tout-un (`setAll()`) sans tenir compte du nombre réel d'octets valides sur le dernier battement.
- **Conséquence** :
  Si un tenseur n'a pas une taille en octets multiple exact de la largeur du bus AXI (ex: 5 octets sur un bus 64 bits), le dernier battement corrompt les 3 octets adjacents en mémoire avec des données poubelles.

---

### BUG-MEM-04 : Deadlock systématique en écriture sur `BramAdapter`, `SramAsicAdapter` et `DdrAdapter`
- **Statut** : corrigé (2026-09-15) — les deux adapters on-chip implémentent un esclave AXI4 write single-outstanding (AW accepté en priorité, beats W commités avec leurs strobes via `Mem.write(mask=...)`, une réponse B), et `UartSoC` connecte désormais les canaux `aw`/`w`/`b` complets de l'accélérateur à la mémoire (avant : seuls les handshakes, payloads non pilotés). `DdrAdapter` arbitre maintenant le port hôte (prioritaire dès qu'une écriture est latée) et le burst accélérateur sur le maître externe : le burst possède le bus de son AW jusqu'à sa réponse B, `aw.len`/`last` sont générés depuis le compteur de beats, `w.strb` est transmis, et la réponse B est routée vers l'accélérateur (auto-acquittée pour l'hôte). Tests rouges : `test_memory_adapter.py::test_bram_adapter_write_burst` / `test_sram_adapter_write_burst` (avant : `aw.ready` bloqué bas ; après : burst relu bit-exact), `test_bram_adapter_write_strobes`, et `test_ddr_adapter_accel_write` (priorité hôte vérifiée : `io_axi_aw_ready` reste bas pendant l'écriture hôte ; puis burst 2 beats avec `len`/`last`/`strb` corrects et `B` forwardé). Formel `AxiReadMemFormal` : write path neutralisé en constantes pour l'élaguer + solveur **boolector** (150 s+ / TIMEOUT → 10 s). Non-régression : `test-all-python -k memory_adapter`, `test-all -k MemoryAdapterTest`, `-k UartSoCTest`. Complément bancs : `MemoryAdapterTest` pilote maintenant explicitement les entrées AXI write (`aw/w.valid := false`, `b.ready := true`) — sans quoi les entrées laissées libres déclenchaient un AW fantôme aléatoire (bancs flaky 2/5) maintenant que `aw.ready` est actif au repos.
- **Fichier** : `spinalML/src/spinalML/memory/BramAdapter.scala`, `SramAsicAdapter.scala`, `DdrAdapter.scala`
- **Code concerné** :
  ```scala
  io.axi.aw.ready := False
  io.axi.w.ready  := False
  ```
- **Description** :
  Les adaptateurs mémoire AXI assignent en dur `ready := False` sur les canaux d'écriture AXI `aw` et `w`.
- **Conséquence** :
  Si l'accélérateur est configuré avec l'option `writeToDdr = true` pour écrire ses résultats via `DMAWriter`, le maître AXI reste bloqué indéfiniment sur la première écriture.

---

### BUG-MEM-05 : Underflow d'adresse et clamping erroné dans `BramAdapter.mapIndex`
- **Statut** : corrigé (durcissement, 2026-09-15) — `mapIndex` clampe désormais les adresses sous `imgBase` sur le **premier** mot physique (`Mux(addr < imgBase, U(0), addr - imgBase)`) au lieu de laisser l'underflow non signé déborder vers le dernier mot ; le clamp haut (au-delà de la mémoire → dernier mot) est conservé. Appliqué aussi à `SramAsicAdapter`. Test rouge : `test_memory_adapter.py::cocotb_bram_adapter_clamp` mis à jour (`addr=0` renvoyait `0xDEAD…` = dernier mot, attend maintenant le premier mot) ; Python 12/12, `MemoryAdapterTest` ✅, formel `AxiReadMemFormal` ✅ (boolector).
- **Fichier** : `spinalML/src/spinalML/memory/BramAdapter.scala` (lignes 44-55)
- **Code concerné** :
  ```scala
  val offset = Mux(isWeight, addr - weightBase, addr - imgBase)
  ...
  Mux(wordIndex >= memoryWords, U(memoryWords - 1, ...), wordIndex.resized)
  ```
- **Description** :
  Si une adresse `addr < imgBase` (par exemple `0x0000_0000`) est présentée, la soustraction non signée `addr - imgBase` subit un underflow et devient une valeur gigantesque ($\sim 0xFFFF\_....$).
- **Conséquence** :
  La condition `>= memoryWords` est satisfaite, et l'accès est mappé sur le tout dernier mot de la BRAM (`memoryWords - 1`). Toute adresse sous la base lit ou écrit silencieusement le dernier mot de la mémoire au lieu de lever une alerte.

---

### BUG-MEM-06 : Interblocage d'ordonnancement des consommateurs dans `TapBuffer.fork`
- **Statut** : faux positif (contrat, vérifié le 2026-09-15) — le chaînage est documenté dans `TapBuffer.scala` (tee atomique : un handshake d'entrée alimente les deux sorties, FIFO à capacité pleine taille) et `TapBufferTest` draine explicitement le consommateur direct avant le différé. Le « deadlock » décrit n'arrive que si un consommateur aval dépend d'un consommateur amont qui ne draine pas (usage hors contrat) ; les usages réels (`Sequential`, `DagDSL.residual`) respectent l'ordre. Repro possible : `direct.ready=0` + `deferred.ready=1` → starvation par construction. Un redesign par FIFO indépendante par branche relèverait de l'évolution, pas du correctif.
- **Fichier** : `spinalML/src/spinalML/memory/TapBuffer.scala` (lignes 59-71)
- **Description** :
  Le chaînage en cascade impose que le consommateur $0$ lise ses données avant le consommateur $1$, qui doit lire avant le consommateur $2$.
- **Conséquence** :
  Si le flux d'exécution matériel fait qu'une couche connectée au consommateur $1$ requiert des données alors que le consommateur $0$ est en attente ou n'a pas fini de consommer, le pipeline se fige en interblocage complet.

---

## 4. Fonctions d'Activation & Poolings

### BUG-ACT-01 : Désynchronisation de séquence dans `MaxPool1D` et `AvgPool1D` sur résidus de pas
- **Fichier** : `spinalML/src/spinalML/poolings/maxpool1d.scala` & `spinalML/src/spinalML/poolings/avgpool1d.scala`
- **Description** :
  Lorsque $(L - poolSize) \pmod{stride} \ne 0$, il reste des éléments en queue de séquence non consommés par la dernière fenêtre de pooling. L'état `stateDone` reboucle directement vers `stateFill` sans purger ces échantillons résiduels.
- **Conséquence** :
  La séquence d'inférence suivante absorbe la queue de la séquence précédente en tête de son propre flux, provoquant un décalage temporel permanent sur toutes les inférences suivantes.

---

### BUG-ACT-02 : `Softmax1D` - Risque de saturation de la table exp
- **Fichier** : `spinalML/src/spinalML/activations/softmax.scala`
- **Description** :
  Le calcul de soustraction du maximum ($x_i - x_{max}$) atténue les débordements positifs, mais pour des écarts importants négatifs, l'exponentielle peut sous-dévier brutalement à 0, rendant la somme des dénominateurs nulle.

---

### BUG-ACT-03 : Désynchronisation spatiale dans `MaxPool2D` et `AvgPool2D` sur dimensions non multiples du pas
- **Fichier** : `spinalML/src/spinalML/poolings/maxpool2d.scala` & `spinalML/src/spinalML/poolings/avgpool2d.scala`
- **Description** :
  Lorsque $(W - K) \pmod{stride} \ne 0$ ou $(H - K) \pmod{stride} \ne 0$, la grille de pooling n'atteint pas le bord droit ou le bord bas de l'image (pixels de queue / résidus de bordure).
  Dès que la dernière fenêtre de sortie est émise, `outCnt.willOverflowIfInc` s'active et la FSM transite vers `stateDone`, qui remet à zéro tous les compteurs (`xCnt`, `yCnt`, etc.) et retourne immédiatement à `stateFill`.
- **Conséquence** :
  Les pixels résiduels de la fin de l'image 1 restent dans le flux d'entrée. `stateFill` commence alors à les consommer en croyant qu'il s'agit du début (pixel (0,0)) de l'image 2 ! Toutes les images suivantes subissent un décalage spatial destructeur.
- **Correction recommandée** :
  Avant de retourner à `stateFill`, purger le reste de l'image jusqu'à ce que $xCnt = W-1$ et $yCnt = H-1$, ou imposer par un `require` que $(W - K) \pmod{stride} == 0$ et $(H - K) \pmod{stride} == 0$.

---

## 5. Couches de Haut Niveau & Attention

### BUG-LAY-01 : Verrouillage permanent du rechargement des poids dans `BatchNorm1D` et `LayerNorm1D`
- **Fichier** : `spinalML/src/spinalML/layers/batchnorm.scala` & `spinalML/src/spinalML/layers/layernorm.scala`
- **Description** :
  La machine d'états charge les coefficients $\gamma$ et $\beta$ lors des états 0 et 1, puis bascule à l'état 2 (inférence) et n'en ressort jamais. Il n'existe aucun compteur de trame ni de port `reArm`.
- **Conséquence** :
  Lors d'une seconde inférence, si le système tente de réacheminer des poids vers `gamma` ou `beta`, `stream.ready` reste à `False`, gelant définitivement le DMA de poids.

---

### BUG-LAY-02 : Incohérence de forme 1D vs 2D pour les poids de `BatchNorm1D` et `LayerNorm1D`
- **Fichier** : `spinalML/src/spinalML/nn/LayerSpec.scala` (lignes 130-137) vs `layers/batchnorm.scala`
- **Description** :
  Dans `LayerSpec.scala`, `getWeightShape()` et `getBiasShape()` retournent `Seq(features, 1)` (2D).
  Dans le composant matériel `batchnorm.scala`, les ports `gamma` et `beta` attendent un tenseur 1D de forme `Seq(channels)`.
- **Conséquence** :
  Risque d'échec de vérification de forme lors de la connexion directe ou de l'inférence formelle de tenseurs.

---

### BUG-LAY-03 : Omission de propagation du signal `reArm` vers `bias_add` dans `Conv1D` et `Conv2D`
- **Statut** : corrigé — `bias_add(..., reArm = Some(io.reArm))` dans `Conv1DLayer` et `Conv2DLayer` (parité avec `Linear.scala:67`). Un reArm en cours de chargement du biais laissait `loadCounter` à mi-course : le biais suivant était chargé décalé (table partiellement périmée). Tests : `tests/python/test_conv1d.py::test_pytest_conv1d_rearm_bias` (nouveau top `Conv1DReArmTestComp`, rouge avant : second beat de biais jamais accepté) ; non-régression Python Conv1D+Conv2D 17/17 et Scala `Conv1DTest` 10/10, `Conv2DTest` 9/9.
- **Fichier** : `spinalML/src/spinalML/layers/Conv1D.scala` (ligne 49) & `spinalML/src/spinalML/layers/Conv2D.scala` (ligne 51)
- **Code concerné** :
  ```scala
  // Conv2D:
  val biasAdded = bias_add(matmulResult, io.b) // reArm non propagé !
  // Conv1D:
  val biased = bias_add(matmulResult, io.b) // reArm non propagé !
  ```
- **Description** :
  Dans `Linear.scala`, le signal `reArm` est explicitement transmis au bloc `bias_add` (`bias_add(..., reArm = Some(io.biasReArm))`) pour réinitialiser le cache de biais et le compteur de trame à chaque commande. Dans `Conv1D` et `Conv2D`, `reArm` est transmis au `matmul` mais omis pour `bias_add`.
- **Conséquence** :
  En cas d'abandon de commande ou de réarmement précoce, le bloc d'addition de biais ne réinitialise pas son état interne et conserve un alignement potentiellement corrompu pour l'inférence suivante.
- **Correction recommandée** :
  Transmettre `reArm = Some(io.reArm)` lors de l'appel à `bias_add` dans `Conv1DLayer` et `Conv2DLayer`.

---

### BUG-LAY-04 : Annulation de l'epsilon par underflow dans `LayerNorm1D` (division par zéro sur FP8/FP4/entiers)
- **Fichier** : `spinalML/src/spinalML/layers/layernorm.scala` (lignes 180-189)
- **Code concerné** :
  ```scala
  case f: FloatML => 
    val epsBits = B(spinalML.utils.MathLUTs.floatEncodeFn(f.expBits, f.mantBits)(1e-5), f.getBitsWidth bits)
  case s: SInt => S(0, s.getWidth bits)
  case u: UInt => U(0, u.getWidth bits)
  ```
- **Description** :
  1. Pour les types entiers `SInt` et `UInt`, `epsVal` est fixé en dur à `0`.
  2. Pour les types flottants compacts comme `FP8_E4M3` (dont la plus petite valeur normale positive est $2^{-6} = 0.015625$), la valeur constante $10^{-5} = 0.00001$ est inférieure de 3 ordres de grandeur au seuil minimal. L'encodage `floatEncodeFn` subit un underflow et encode cette valeur sous forme de `0.0`.
- **Conséquence** :
  Pour `SInt`, `UInt`, `FP8` et `FP4`, $\epsilon$ vaut réellement zéro. Si tous les éléments d'un vecteur d'entrée sont identiques (ex. vecteur nul ou constant), la variance $\sigma^2$ vaut $0$, et l'opération évalue $\frac{1}{\sqrt{0 + 0}} = \frac{1}{0}$ (division par zéro et saturation ou crash du bloc `rsqrt`).
- **Correction recommandée** :
  Choisir un $\epsilon$ adapté à la dynamique du format (par exemple la plus petite valeur représentable positive du type `FloatML` : $2^{1-bias}$, et `1` pour les entiers).

---

### BUG-LAY-05 : Troncature arithmétique sans saturation dans `BatchNorm1D` sur entiers `SInt`
- **Fichier** : `spinalML/src/spinalML/layers/batchnorm.scala` (lignes 46-49)
- **Code concerné** :
  ```scala
  case (vx: SInt, va: SInt, vb: SInt) =>
    outPayload(i).assignFrom(((vx * va) + vb).resized.asInstanceOf[T])
  ```
- **Description** :
  Le produit $vx \times va$ s'étend sur 16 bits. L'application directe de `.resized` vers le type de sortie 8 bits tronque les bits de poids fort sans décalage en virgule fixe (pas de division par $2^Q$) et sans saturation.
- **Conséquence** :
  Tout produit dépassant $[-128, +127]$ subit un bouclage en complément à deux (modulo 256), inversant les signes et corrompant les résultats pour les réseaux de neurones quantifiés en entiers.
- **Correction recommandée** :
  Ajouter un paramètre de décalage (`shift`) et un clamp/saturation symétrique.

---

## 6. Réseau Séquentiel, DAG & Accélérateur

### BUG-NN-01 : Violation du protocole AXI/Stream et boucle combinatoire sur les requêtes de poids
- **Fichier** : `spinalML/src/spinalML/nn/Sequential.scala` (lignes 381-383 et 470-471)
- **Code concerné** :
  ```scala
  reqW.valid := startPathW ||
    (prefetchWorldW && (reloadPendingW || residentRise) &&
      reqW.ready && wDoubleBuffer.io.loadCanAccept && !startPathW)
  ```
- **Description** :
  La condition d'émission `valid` incorpore directement le signal `reqW.ready` de l'esclave.
- **Conséquence** :
  Violation stricte des standards AXI et SpinalHDL (`valid` ne doit jamais dépendre combinatoirement de `ready`). Cela crée une boucle de rétroaction combinatoire dès que l'esclave fait dépendre son `ready` de `valid`.

---

### BUG-NN-02 : Écrasement et extinction prématurée de `ioBusy` lors de trames consécutives
- **Fichier** : `spinalML/src/spinalML/nn/Sequential.scala` (lignes 710-715)
- **Code concerné** :
  ```scala
  when(io.start.fire) {
    ioBusy := True
  }
  when(ioBusy && io.outStream.stream.fire && frameCounter.willOverflowIfInc) {
    ioBusy := False
  }
  ```
- **Description** :
  En Scala/SpinalHDL, la dernière affectation l'emporte. Si `io.start.fire` (démarrage d'une nouvelle inférence) survient au même cycle que l'émission du dernier battement de l'inférence précédente, la deuxième condition écrase la première.
- **Conséquence** :
  `ioBusy` repasse immédiatement à `False`, faussant le signal de statut de l'accélérateur qui se déclare inactif alors qu'une inférence vient de commencer.

---

### BUG-NN-03 : Découpage de flux poids bloquant dans `ClassicalAttention`
- **Fichier** : `spinalML/src/spinalML/nn/Sequential.scala` (lignes 665-679)
- **Code concerné** :
  ```scala
  val wForks = StreamFork(layerWeights.stream, 4)
  ...
  comp.io.wq <> spinalML.ops.slice(w0, 0, a.embedDim, axis = 0)
  comp.io.wk <> spinalML.ops.slice(w1, a.embedDim, 2 * a.embedDim, axis = 0)
  comp.io.wv <> spinalML.ops.slice(w2, 2 * a.embedDim, 3 * a.embedDim, axis = 0)
  comp.io.wo <> spinalML.ops.slice(w3, 3 * a.embedDim, 4 * a.embedDim, axis = 0)
  ```
- **Description** :
  `StreamFork` synchronise 4 branches de découpage séquentiel sur un même flux unifié. Comme `SliceAxis0Op` utilise un compteur sans remise à zéro (`Counter(L_in)`), si une désynchronisation d'un seul battement survient entre les consommateurs $W_q, W_k, W_v, W_o$, le fork bloque l'ensemble du flux de poids.

---

### BUG-NN-04 : Absence de garde-fou sur `dmaCmd.length` dans `Accelerator.scala` (risque de troncature 16 bits)
- **Statut** : faux positif (vérifié le 2026-09-15) — `U(valeur, width)` **lève une exception** si la valeur ne tient pas dans la largeur (`core/internals/Expression.scala:2457`, « literal ... can't fit in UInt(...) ») : il n'y a pas de troncature silencieuse. Un `require(totalOutBeats <= 65536 && > 0)` reste une amélioration de message d'erreur possible (non appliquée).
- **Fichier** : `spinalML/src/spinalML/nn/Accelerator.scala` (ligne 164)
- **Code concerné** :
  ```scala
  val totalOutBeats = (model.finalShape.product + outLanesAxi - 1) / outLanesAxi
  ...
  dmaCmd.length := U(totalOutBeats - 1, 16 bits)
  ```
- **Description** :
  Dans `Sequential.scala`, la taille de commande DMA est explicitement bornée par `require(beats <= 65536)`. Dans `Accelerator.scala`, aucun `require` ne protège `totalOutBeats`.
- **Conséquence** :
  Si un modèle possède une sortie large (supérieure à 65 536 battements, soit $> 512$ Ko sur bus 64 bits), `totalOutBeats - 1` déborde 16 bits et est tronqué silencieusement lors de l'instanciation de `U(..., 16 bits)`. Seule une fraction des données de sortie sera écrite en mémoire DDR.
- **Correction recommandée** :
  Ajouter `require(totalOutBeats <= 65536 && totalOutBeats > 0)` à l'élaboration.

---

### BUG-NN-05 : Absence de réinitialisation logicielle de `imgBaseOffset` dans `Accelerator.scala`
- **Statut** : corrigé — `ctrlFactory.onWrite(0x08) { imgBaseOffset := 0 }`, placé après l'incrément `frameDone` pour qu'une écriture hôte gagne en cas de collision dans le même cycle. Une écriture 0x08 repart donc du nouveau base sans hard reset. Tests : `tests/python/test_accelerator.py::test_accel_image_base_reset` (rouge avant : zéros lus à `new_base + k*FRAME_BYTES`) ; non-régression `test_accelerator.py` 5/5 (dont RUN continu).
- **Fichier** : `spinalML/src/spinalML/nn/Accelerator.scala` (lignes 221-233)
- **Code concerné** :
  ```scala
  val imgBaseOffset = Reg(UInt(axiConfig.addressWidth bits)) init(0)
  ...
  when(frameDone) {
    tileCntReg := tileCntReg + 1
    when(runActive) {
      startPending := True
      imgBaseOffset := imgBaseOffset + imageBytesAcc
    }
  }
  ...
  model.io.imgBaseAddress := imgAddrReg + imgBaseOffset
  ```
- **Description** :
  En mode continu (`RUN`), `imgBaseOffset` avance de `imageBytesAcc` à chaque image. Cependant, aucun mécanisme logiciel ne permet de remettre ce décalage à 0 (ni l'écriture d'une nouvelle adresse dans CSR `0x08`, ni la désactivation/réactivation du mode `RUN`).
- **Conséquence** :
  Si l'hôte exécute une première série de 10 images, puis souhaite relancer une nouvelle série en écrivant une nouvelle adresse de base dans `0x08`, l'accélérateur commence à lire à `nouvelle_adresse + 10 * imageBytesAcc`, corrompant les adresses de lecture pour toutes les sessions suivantes sans hard reset du FPGA.
- **Correction recommandée** :
  Réinitialiser `imgBaseOffset := 0` lors d'une écriture sur `0x08` (`ctrlFactory.onWrite(0x08) { imgBaseOffset := 0 }`) ou sur front montant de `runActive`.

---

### BUG-NN-06 : Danger de blocage en polling CPU sur le bit de complétion CSR `0x04`
- **Statut** : corrigé — registre verrou `doneSticky` mis à 1 sur `frameDone` et remis à 0 uniquement sur écriture hôte de 0x00 ; CSR 0x04 bit 0 lit `Mux(writeToDdr, doneSticky, outStream.valid)`. Un hôte qui poll ne peut plus rater l'impulsion d'un cycle (la lecture AXI-Lite dure elle-même plusieurs cycles). Tests : `tests/python/test_accelerator.py::test_accel_dma_write_back` (assertion ajoutée : 50 cycles après `io_done`, lire 0x04 et exiger bit0=1 ; rouge avant fix) ; non-régression Python 5/5 + Scala `AcceleratorTest` 4/4.
- **Fichier** : `spinalML/src/spinalML/nn/Accelerator.scala` (ligne 239)
- **Code concerné** :
  ```scala
  ctrlFactory.read(Mux(writeToDdr, dmaWriter.io.done, io.outStream.stream.valid), 0x04, 0)
  ```
- **Description** :
  En mode écriture DDR (`writeToDdr = true`), le bit 0 du registre de statut `0x04` reflète directement `dmaWriter.io.done`. Or, `dmaWriter.io.done` est une **impulsion de seulement 1 cycle d'horloge**.
- **Conséquence** :
  Une transaction de lecture AXI-Lite depuis un CPU hôte prend généralement entre 10 et 50 cycles d'horloge. La probabilité d'effectuer la lecture au cycle exact de l'impulsion est quasi nulle. Un pilote logiciel qui boucle en attendant `bit 0 == 1` ne verra jamais la fin de l'inférence et partira en timeout.
- **Correction recommandée** :
  Remplacer l'impulsion par un registre verrou (latch/sticky bit) qui passe à 1 sur `frameDone` et se remet à 0 au déclenchement de `startPending` ou par lecture.

---

## 7. Entrées / Sorties, Pont Série & SoC

### BUG-IO-01 : Perte silencieuse d'octets entrants dans `UartBridge` (`rx.ready := True` permanent)
- **Statut** : faux positif (mécanisme erroné, vérifié le 2026-09-15) — `UartRx` n'expose **aucun `ready`** (sa `valid` est une impulsion d'un cycle) et `UartSoC` ne connecte pas `bridge.io.rx.ready` au récepteur (`io.rx.valid`/`payload` seulement) : gater `io.rx.ready` ne changerait donc rien. La perte n'existe que si l'hôte pipeline des commandes, ce que le protocole requête/réponse du top.v de référence interdit. Un durcissement réel (FIFO RX + `ready` honoré) serait une évolution hors périmètre.
- **Fichier** : `spinalML/src/spinalML/io/UartBridge.scala` (ligne 101)
- **Code concerné** :
  ```scala
  io.rx.ready := True
  ```
- **Description** :
  Le signal `io.rx.ready` est maintenu à `True` en continu, y compris dans les états où la FSM est occupée à dialoguer sur le bus CSR (`C_EXEC_AW`, `C_EXEC_B`) ou à émettre des données sur l'UART (`R_WAIT`, `R_SEND`, `S_SEND`).
- **Conséquence** :
  Tout octet envoyé par l'hôte pendant ces états est immédiatement acquitté et jeté sans être lu, corrompant les trames de commandes successives.
- **Correction recommandée** :
  N'activer `io.rx.ready` que dans les états récepteurs :
  ```scala
  io.rx.ready := (state === IDLE) || (state === C_ADDR) || (state === C_VAL) ||
                 (state === W_ADDR) || (state === W_LEN) || (state === W_DATA)
  ```

---

### BUG-IO-02 : Corruption de données lors d'écritures BRAM partielles dans `UartBridge`
- **Statut** : corrigé (2026-09-15) — `UartBridge` expose désormais un masque d'octets `wrStrb` (bit i = octet i valide) laté par mot : tout-1 pour un mot complet, seuls les octets reçus pour le dernier mot partiel. `MemoryAdapter.io` (donc `BramAdapter`/`AxiReadMem`/`SramAsicAdapter`) reçoit `wrStrb` et le convertit en masque bit-à-bit pour `Mem.write(mask=...)` ; `DdrAdapter` le recopie dans `w.strb` AXI. Test rouge : `test_uart_bridge.py::test_uart_bridge_partial_write` (mot complet 0xAA puis 3 octets au mot suivant : avant fix la queue valait `0xAA…`, après fix elle reste à zéro) et `test_memory_adapter.py::test_bram_adapter_partial_write` (strobes 0x07, octets hauts préservés). Non-régression : `test-all-python -k uart` 3/3, `-k memory_adapter`, `test-all -k MemoryAdapterTest/UartBridgeTest/UartSoCTest`, formels `AxiReadMemFormal` (+ `anyseq(wrStrb)`) et `UartBridgeFormal`.
- **Fichier** : `spinalML/src/spinalML/io/UartBridge.scala` (lignes 226-240)
- **Code concerné** :
  ```scala
  wrBytes(byteCnt.resize(log2Up(wordBytes) max 1)) := io.rx.payload
  lenReg := lenReg - 1
  when(byteCnt === (wordBytes - 1) || lenReg === 1) {
    wrEnableR := True
    byteCnt := 0
    ...
  }
  ```
- **Description** :
  `wrBytes` est un vecteur de registres sans masquage d'octets. Si la longueur totale d'écriture `len` n'est pas un multiple exact de la largeur du mot mémoire (ex. écriture de 10 octets avec un bus de 64 bits = 8 octets), le deuxième mot écrit 2 octets valides suivis de 6 octets obsolètes résiduels du premier mot.
- **Conséquence** :
  Corruption silencieuse des octets mémoires suivant immédiatement une charge utile non alignée.

---

### BUG-IO-03 : Violation du protocole Stream sur `io.tx.valid` (impulsion d'un seul cycle)
- **Statut** : faux positif (vérifié le 2026-09-15) — la FSM ne pulse `txStartR` que dans `R_WAIT` (condition `io.outStream.valid && io.tx.ready`) ou en `IDLE`, atteint uniquement après que `tx.ready` soit repassé haut en `R_SEND`/`S_SEND` ; les commandes C/W ne touchent pas TX. De plus `UartTx` est une interface `start`+`ready` (pas un Stream) : l'impulsion d'un cycle n'est pas une violation et aucun octet n'est perdu.
- **Fichier** : `spinalML/src/spinalML/io/UartBridge.scala` (lignes 102, 123)
- **Code concerné** :
  ```scala
  txStartR := False // défaut
  ...
  io.tx.valid := txStartR
  ```
- **Description** :
  `txStartR` est pulsé pendant exactement 1 cycle d'horloge. Si `io.tx.ready` n'est pas haut à cet instant précis (par exemple si l'émetteur UART est en train de clore un bit de stop), `valid` retombe à zéro sans que l'octet n'ait été transféré.
- **Conséquence** :
  Violation du contrat Stream de SpinalHDL et perte pure et simple d'octets de sortie vers le PC hôte.

---

### BUG-IO-04 : Tronquage statique d'adresse CSR dans `UartBridge`
- **Statut** : corrigé — le mode de défaillance réel n'est pas une troncature silencieuse mais une **erreur d'élaboration** (`WIDTH MISMATCH`) dès que `csrAddrWidth != 8` : `addrReg(7 downto 0)` (8 bits) est assigné à `csrAwAddrR` de largeur `csrAddrWidth`. Fix : `csrAwAddrR := addrReg.resize(csrAddrWidth)`. Tests : nouveau top `UartBridgeWideCsrTestComp` (csrAddrWidth = 12, généré par `UartBridgeTest.scala`) + `tests/python/test_uart_bridge.py::test_uart_bridge_wide_csr` (C-commande vers 0x123, moniteur AW ; rouge avant par échec d'élaboration) ; non-régression `test_uart_bridge.py` 2/2 et Scala `UartSoCTest` 2/2.
- **Fichier** : `spinalML/src/spinalML/io/UartBridge.scala` (ligne 169)
- **Code concerné** :
  ```scala
  csrAwAddrR := addrReg(7 downto 0)
  ```
- **Description** :
  L'adresse est tronquée brutalement aux 8 bits de poids faible, ignorant le paramètre générique `csrAddrWidth`.
- **Conséquence** :
  Incapacité d'adresser des registres au-delà de `0xFF` si un accélérateur étend son plan mémoire CSR.

---

### BUG-IO-05 : Exception d'élaboration sur largeurs de types $> 8$ bits dans `UartSoC`
- **Statut** : corrigé (garde d'élaboration, 2026-09-15) — `UartSoC` vérifie en tête d'élaboration `outElemBits == 8` et échoue avec un message explicite (« UartSoC only serializes 8-bit output elements over UART (got N bits) — see docs/uart_bridge.md ») au lieu du WIDTH MISMATCH cryptique du sérialiseur ; le protocole `R` reste 1 octet/logit (FP8/INT8). Le support BF16/FP16 est tracé par `TODO(multi-byte-logits)` dans `UartSoC.scala` et une entrée backlog dans `docs/full_roadmap.md` (Refactoring 1.4 : sérialisation `outCount × bytesPerElem`, mise à jour de `uart_host.read_logits` et de la doc). Test : `UartSoCTest` « refuses non-8-bit output elements » (BF16 → exception contenant « 8-bit ») ; `UartSoCTest` 3/3, `MemoryAdapterTest` ✅.
- **Fichier** : `spinalML/src/spinalML/io/UartSoC.scala` (lignes 109-118)
- **Code concerné** :
  ```scala
  def toByte(elem: Data): Bits = elem match {
    case f: FloatML => (f.sign ## f.exponent ## f.mantissa).asBits
    case b          => b.asBits.resize(8)
  }
  ```
- **Description** :
  Pour un type flottant 16 bits (ex. `BF16` ou `FloatML(5, 10)`), `toByte` produit un signal de 16 bits, qui est ensuite assigné au port `bridge.io.outStream.payload` typé en `Bits(8 bits)`.
- **Conséquence** :
  Échec critique à la compilation SpinalHDL (`assignment width mismatch: 16 bits vs 8 bits`).

---

### BUG-IO-06 : Compteur d'octets `byteCnt` sous-dimensionné (3 bits) pour des bus de données $> 64$ bits dans `UartBridge`
- **Statut** : corrigé — bug réel, mode de défaillance observé : **erreur d'élaboration** `OUT OF RANGE CONSTANT` (`byteCnt === wordBytes - 1`, ex. 15 pour un bus 128 bits) dès que `wordWidth > 64`, pas de corruption silencieuse. Fix : largeur calculée à l'élaboration `Reg(UInt(log2Up((wordBytes max 4) + 1) bits))` (le compteur sert aussi aux champs 4 octets des commandes) — aucune limite en dur, fonctionne pour 64/128/256 bits. Test : nouveau top `UartBridgeWideWordTestComp` (wordWidth = 128) + `tests/python/test_uart_bridge.py::test_uart_bridge_wide_word` (W de 16 octets, mot 128 bits relu ; rouge avant par échec d'élaboration) ; non-régression `test_uart_bridge.py` 3/3, Scala `UartBridgeTest` 3/3 + `UartSoCTest` 2/2.
- **Fichier** : `spinalML/src/spinalML/io/UartBridge.scala` (lignes 77, 226-239)
- **Code concerné** :
  ```scala
  val byteCnt = Reg(UInt(3 bits)) init 0
  ...
  when(byteCnt === (wordBytes - 1) || lenReg === 1) { ... }
  ```
- **Description** :
  Le registre `byteCnt` est alloué avec une largeur codée en dur de 3 bits (capacité de comptage de 0 à 7).
  Si la passerelle `UartBridge` est instanciée avec une largeur de mot mémoire BRAM supérieure à 64 bits (par exemple `wordWidth = 128` bits, soit `wordBytes = 16`, ou `wordWidth = 256` bits, classique sur les interconnexions AXI haut débit), `byteCnt` boucle de 0 à 7 sans jamais pouvoir atteindre `wordBytes - 1` (15 ou 31).
- **Conséquence** :
  Pour tout bus de données plus large que 64 bits, la condition de fin de mot `byteCnt === (wordBytes - 1)` n'est jamais satisfaite. Les octets supérieurs de `wrBytes` ne sont jamais assignés, et l'impulsion d'écriture `wrEnableR` n'est émise qu'en toute fin de transfert (`lenReg === 1`), corrompant l'ensemble des écritures mémoire BRAM.
- **Correction recommandée** :
  Dimensionner dynamiquement `byteCnt` selon le maximum entre la taille d'adresse/longueur (4 octets) et la taille d'un mot (`wordBytes`) :
  ```scala
  val byteCnt = Reg(UInt(log2Up((wordBytes max 4) + 1) bits)) init 0
  ```

---

## 8. Utilitaires & DSL

### BUG-DSL-01 : Indexation erronée des connexions résiduelles dans `DagDSL.residual`
- **Statut** : faux positif (vérifié le 2026-09-15) — le contrat documenté (`DagDSL.scala:8-18`) exige `base` = nœud d'entrée = fin courante au moment de l'appel, et l'unique usage (`tests/universal/UniversalTransformerDemo.scala:26,31`) le respecte. La formule `base + branch.length` est correcte sous ce contrat ; le risque n'existe que si un appelant passe un `base` antérieur à la fin courante (usage hors contrat).
- **Fichier** : `spinalML/src/spinalML/nn/DagDSL.scala` (ligne 31)
- **Code concerné** :
  ```scala
  branch ++ Seq(Add(a = base, b = base + branch.length))
  ```
- **Description** :
  La formule calcule le second nœud sous la forme `b = base + branch.length`. Cela présuppose obligatoirement que la branche commence immédiatement à l'indice `base + 1`.
  Si la connexion résiduelle saute par-dessus des couches préalablement insérées (par exemple un skip-connection depuis l'entrée `node 0` par-dessus un bloc inséré plus tôt), `b` pointe vers une mauvaise couche interne au lieu de pointer vers la fin de la branche nouvellement créée.
- **Conséquence** :
  Construction d'une topologie de réseau corrompue sommant des nœuds non désirés.

---
*Fin du rapport d'audit post-refactoring.*
