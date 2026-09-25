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

## Dérivations à venir (définies à la lecture de chaque zone)

- **4b `nn` spill tests** : à définir.
- **4c `arithmetic`/`ops` RTL** : à définir (pressenti : largeurs /
  latence / reset en 1 ligne par branche).
- **4d `layers`/`memory`** : à définir.
- **4e `harness`/`examples`/`dtypes`** : à définir.
- **4f Python CLI (`cli/`)** : à définir (même esprit, syntaxe `#`).
- **4g Python `scripts/` + `tests/python`** : à définir.
