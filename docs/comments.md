# Conventions commentaires spinalML

Principe : un commentaire dit ce que le code **ne dit pas** (*pourquoi*,
contrat, unités, edge case). **Ce qui est déjà bon ne se réécrit pas**
(diff minimal : purge = suppression + compression, jamais reformulation).

## Socle universel (tout fichier Scala, toute zone)

- Copyright d'en-tête gardé tel quel.
- `/** */` : objets/classes/méthodes publiques, ≤ 5 lignes
  (quoi + contrat + pointeur). Privé : 1 ligne si non-évident, rien sinon.
- Inline `//` : 1 ligne, le *pourquoi*. Formules, unités et edge cases
  bienvenus ; paraphrase du code interdite.
- Interdits : bannières ASCII (`// --- X ---`), code commenté,
  `@author`/`@date`, TODO sans contexte.
- Ancres : référence par nom de symbole (`cf. sliceTransposeFlat`),
  jamais « ci-dessus ».
- Unités : dans les noms (`...Bytes`, `...Bits`), pas en prose répétée.
- Debug : `SimLog` uniquement (`debug`/`trace`, tags majuscules), zéro
  `println` nu sauf verdicts (`PASSED`, `DEV`) et outils CLI.
- Tags de session interdits (`Wave 5`, `DTYPE-xx`, `LAY-xx`, …) :
  la leçon s'écrit en clair en 1 ligne, la provenance vit dans git.
- Commentaires autoporteurs : aucun pointeur vers `docs/` (appelée à
  partir en archive) — l'essentiel tient en 1 ligne inline. Seuls les
  pointeurs **code** survivent (`mirrors RTL <symbole>`, `cf. <f>`).

## Dérivation `replica` (zone 4a, validée)

- **R1 formules** : tout ordre mémoire en 1 ligne canonique
  (`p*Ks*N + n*Ks + k_local`, axes `[C][H][W]` sur les params) ;
  toute répétition = pointeur vers la définition canonique.
- **R2 miroirs** : chaque règle arithmétique cite sa source **code**
  en 1 ligne (`mirrors RTL <X>` / `mirrors spinalML.utils.Float.<Y>`) ;
  l'essentiel des contrats ex-docs est inliné, sans pointeur `docs/`.
- **R3 handlers** : 1 ligne d'objet obligatoire (couches couvertes) ;
  les `require` portent le contrat (inchangés).
- **R4 tests** : class-doc = contrat prouvé ≤ 6 lignes + géométrie
  1 ligne inline.

## Dérivation `nn` e2e/SoC tests (zone 4b, validée)

- **N1 suites e2e** : class-doc = preuve (quoi vs quel oracle) + géométrie
  1 ligne ; labels de phase gardés (repris par les noms de tests).
- **N2 bench discipline** : settle/moniteurs en clair, explications gardées ;
  tags S1/S2b/S2e retirés ; moniteurs cycle-à-cycle déjà migrés SimLog
  TRACE (commit 1).
- **N3 knob/config** : géométrie chiffrée inline gardée ; pointeurs `docs/`
  → essentiel inliné (docs appelées à partir en archive).
- **N4 SoC goldens** : X/W/b + expected inline gardés ; step-markers
  paraphrases (`Pulse START`, `Program addresses`, …) dehors.
- **N5 bug IDs** : `NN-02`, `BUG-DDR-*`, `LAY-02` gardés (noms de
  tests/messages), explications gardées.

## Dérivation `arithmetic`/`ops` RTL (zone 4c, validée)

- **C1 contrats HW** : shapes/lanes/latences/reset et ordres (row-major,
  `[K,K,C]`, Q-formats) en 1 ligne là où le code ne le dit pas.
- **C2 sessions unwrap** : `OPS-02/04/05/07/09/10`, `Option B`, `DTYPE-06`,
  `bisection M1.7` retirés ; règles et gardes expliqués en clair, gardés.
- **C3 anglais partout** : commentaires + messages runtime traduits
  (`cumsum`/`add`/`sub`/`cast` étaient en français).
- **C4 structure** : bannières ASCII dehors ; labels d'algo par branche
  gardés (PWL, algebraic, SPLIT/AGGREGATE) ; step-markers paraphrases
  dehors.

## Dérivation `nn` core RTL (zone 4d, validée)

- **D1 contrats d'élaboration** : requires/knobs/footprints/cursors en
  clair ; pointeurs `docs/` → essentiel inliné.
- **D2 phases unwrap** : `S0/S1/S2x`, `M2/M3`, `Phase-N` retirés ;
  `NN-01` → sticky ; `NN-02` gardé (noms de tests) ; leçons (K64-P8,
  rerun, reArm, gearbox) gardées en clair.
- **D3 anglais partout** : le français restant traduit (commentaires +
  messages).
- **D4 structure** : bannières ASCII → titres nus numérotés ; step-markers
  (`1. Instantiate…`) dehors ; miroirs inter-blocs (`mirror of…`)
  gardés comme navigation.
## Dérivation `layers`/`memory` RTL (zone 4e, validée)

- **E1 contrats de flux** : shapes/lanes/ordres/fences en 1 ligne ;
  formules et invariants (beat-align, sticky, RAW) gardés intacts.
- **E2 sessions unwrap** : `P1/P2/S1/S2`, `M1.7`, `M2`, `Phase-N`,
  `Option B` retirés ; `BUG-DDR-04/07` gardés (noms de tests) ;
  pointeurs `docs/` → essentiel inliné.
- **E3 structure** : bannières ASCII → titres nus (sections numérotées
  gardées) ; step-markers paraphrases dehors ; labels d'algo par branche
  gardés.
- **E4 port docs** : rôles non-évidents gardés (lanes=1, accType,
  reArm), paraphrases (`Input Sequence`, `Kernel Weights`…) dehors.

## Dérivation `src` restant (zone 4f, validée)

- **F1 golden-mirror** (`utils/Float`, `PWL`, `math_luts`, narrow-float
  softmax) : contrats bit-exact, formules, RNE guard/sticky, slots E4M3
  gardés ; tags `Wave 4/5`, `DTYPE-06/07`, `docs/rounding_policy.md`
  unwrappés ; steps numérotés → titres nus.
- **F2 bug IDs test-anchored** : `ACT-01/03` (poolings), `OPS-07`
  (attention/matmul), `OPS-10` (casts) gardés comme `BUG-DDR-*`
  (repris par les noms de tests) ; explications gardées.
- **F3 switch Rne/Truncate** : mentions `[[Rne]]`/`[[Truncate]]` (code)
  gardées ; `Option B`, `switch-aware` dehors ; leçon en clair.
- **F4 examples/templates** : bannières → titres nus ; shapes, domaines
  (int vs float), contrats API gardés ; `Generate the Verilog` dehors ;
  `M2/M3` unwrappés (leçons `copy()`, LUT pole, chunk fold gardées).
- **F5 SoC/IO** : miroirs `top.v`/Verilog gardés ; `Phase-N` unwrappé ;
  pointeurs `docs/` → essentiel inliné ; TODO contextuels gardés.
- **F6 V1** : `V1` en clair dans les commentaires ; strings
  `require`/messages runtime intouchés.
- **Exclusions** : `Target.scala` (clean), `SpinalMLConfig` (string
  généré), tables de poids `Mnist*` (seuls les tags bougent),
  fichiers Copyright-only (`dsp`, `primitives`, `MemLayout`, `SimLog`,
  `UartTx`, `UartChainGen`, `AxiReadMem`, `UartSoCGen`,
  `QuantActivation`).

## Dérivation tests Scala restants (zone 4g, validée)

- **G1 oracles** : expected values/géométrie inline gardés (preuves) ;
  `Step N:` dehors ; `Component for testing X` → géométrie 1 ligne.
- **G2 labels test-anchored gardés** : `OPS-02/04/05/07/10`, `DTYPE-06`,
  `ACT-01/03` (noms de tests) ; tags `Wave N`, `DTYPE-07`, `M1/M2/M3`,
  `S1/S2`, `P0`, `Phase-*` unwrappés — noms de tests et strings
  `require`/`assert`/`SimConfig.withWave` intouchés (code).
- **G3 formal** : class-docs ≤ 5 lignes ; pointeurs `docs/` → essentiel
  inliné ; `mirrors <Op>.<fn>` gardés ; FR traduit.
- **G4 harness** : markers TEST-ONLY gardés ; `Phase-*`/`docs/`
  unwrappés.
- **G5 structure** : bannières `----`/`===` → titres nus ; bench
  discipline (bounded waits, watchdogs) gardée.

## Dérivations à venir (définies à la lecture de chaque zone)

- **4f `src` restant** : fait (F1-F6 ci-dessus).
- **4g tests Scala restants** : fait (G1-G5 ci-dessus).
- **4h Python CLI (`cli/`)** : à définir (même esprit, syntaxe `#`).
- **4i Python `scripts/` + `tests/python`** : à définir.
