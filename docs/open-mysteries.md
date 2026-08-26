# Mystères ouverts — registre des comportements non expliqués

> **Objectif** : un seul endroit où vivre pour tout ce que le projet **ne sait pas encore
> expliquer**, avec les hypothèses classées, les expériences de falsification concrètes et le
> plan de dissection. Un mystère non écrit est un piège garanti pour la phase suivante.
>
> **Statuts** : 🔴 OUVERT · 🟡 EN COURS · 🟢 RÉSOLU (avec cause racine documentée)
>
> **Docs compagnons** : `docs/dataflow-map.md` (carte de communication, abrégée « carte »),
> `docs/bugs/2026-08-rearm-session.md` (rapport RC), `docs/symbolicTestPlaybook.md` (méthodo
> formelle BMC).

---

| ID | Sujet | Statut | Bloque |
|---|---|---|---|
| [M1](#m1--gearbox-structurée-flushable--graphe-dag) | Gearbox structurée flushable × graphe DAG | 🟡→🟢 **CAUSE RACINE RÉGIONALE PROUVÉE** (bisection C0–C8) ; règle d'élasticité au fan-out établie ; micro-mécanisme RTL = suivi optionnel | Ne bloque plus : le cloison img-legacy/poids-flushable est principé (voir M1.7) |
| [M2](#m2--état-de-fenêtre-persistant-dim2colop) | État de fenêtre persistant d'Im2ColOp | 🟡 OUVERT-documenté | Abort/soft-reset futur ; refonte multi-tile Phase 3 |

---

## M1 — Gearbox structurée flushable × graphe DAG

### M1.1 Les faits (évidence accumulée)

**Config qui ÉCHOUE** — gearbox structurée aussi sur le chemin image (session ré-armement,
itérations « single-mode » puis expA2) :

- `ResidualMLPTemplate` ([2,4] BF16, skip connection) produit une **seconde rangée fausse** :
  sortie observée `[1.5, 3.0, 4.5, 6.0, 0.0, 0.0, 0.625, 0.25]` — première moitié correcte,
  puis zéros insérés / valeurs décalées.
- Source : rapport RC §5.2.

**Configs qui MARCHENT** — c'est ce rend le mystère :

| Config | Stalls aval ? | Résultat |
|---|---|---|
| Probes standalone du même split 4→1 (BF16, SInt, FloatML) | NON (consommateurs eager) | bit-parfaits (1024/1024) |
| Gearbox structurée dans readers poids/biais réels | OUI (backpression matmul réelle) | suite Scala 50/50 verte |
| Cloison actuel : image legacy / poids flushable | les deux | chaîné 20/20 + DAG 6/6 + Python 3/3 |

Lecture : la gearbox structurée fonctionne sous stalls quand elle est dans le chemin poids ;
elle casse un DAG quand elle remplace l'adapter legacy **ailleurs** (chemin image et/ou repacks
de graphe entre ops). Le facteur discriminant n'est donc pas la gearbox elle-même mais
**quelque chose d'interaction** : cadence, position dans le graphe, ou type de consommateur.

### M1.2 Ce qui est DÉJÀ établi (et n'est donc PAS le bug)

Analyses statiques faites pendant la session — à convertir en invariants formels (étape B) :

1. **AGGREGATE : le fire croisé entrée/sortie est impossible.**
   `io.a.stream.ready := !full` et `io.c.stream.valid := full` (ops/repack.scala:68-69) sont
   contradictoires sur le même cycle ⇒ jamais simultanéité ⇒ le scénario « collecte écrasée par
   drain » ne peut pas se produire.
2. **SPLIT : la priorité reload-sur-drain-final est correcte par construction.**
   `when(io.a.stream.fire)` prime sur le bloc sinon-drain (ops/repack.scala:48-57) : dernier
   slice drainé + nouveau mot capturé au même cycle = transition propre.
3. **isEmpty agrégé sur les chaînes** : tous les RepackOps créés participent au gate de
   commande (memory/DMAReader.scala:130-134).

MAIS tout ceci reste du raisonnement sur papier : aucun invariant formel n'est prouvé à ce jour.

> **Mise à jour sprint (2026-08)** : les items 1 et 2 sont désormais **prouvés formellement** —
> l'assertion d'exclusion mutuelle est intégrée au spec `RepackStallAggregateFormal`
> (BMC 14 cycles, CVC4) et la discipline de capture du SPLIT est couverte par simulation à stalls
> aléatoires (voir M1.6).

### M1.3 La différence mécanique clé (pourquoi la cadence change)

C'est le cœur du soupçon H1 — les deux gearboxes n'imposent **pas le même rythme amont** :

| | Adapter legacy (lib) | SPLIT structuré (nôtre) |
|---|---|---|
| Mécanique | Découpe **combinatoire** du payload amont, tenu stable (`output.valid := input.valid`, lib Stream.scala:2132-2137) | Capture en registre `hold` dès le premier fire, amont **libéré immédiatement** (ops/repack.scala:43-45) |
| Acceptation beat suivant | Seulement au dernier slice (`input.ready` sur willOverflowIfInc) | Possible dès `!full` (overlap acceptation/drain) |
| Conséquence | L'amont voit des périodes « valid tenu, prêt bas » plus longues ; les groupes sortent avec le pacing exact de l'amont | Les groupes peuvent sortir collés différemment ; les cycles de `valid` aval tombent à d'autres positions relatives |

Un consommateur dont une FSM suppose implicitement « pas de trou ici » ou « ce signal tombe
toujours après celui-là » passe avec l'un et casse avec l'autre.

### M1.4 Hypothèses classées + falsification

| # | Hypothèse | Arguments pour | Expérience qui la tranche | Statut sprint |
|---|---|---|---|---|
| **H1** (favorite) | **Bug latent de pacing chez un consommateur** du DAG (entre im2col et la FIFO tap, wording du rapport RC §5.2) : un op tolère mal le nouveau régime de cadence | Le même hardware marche/échoue selon la seule gearbox amont ; les probes eager ne voyaient pas de stalls | Harnais différentiel **par op** (étape A ci-dessous) : source à stalls aléatoires × {legacy, flushable} → score vs modèle logiciel. L'op qui diverge est le coupable | **Reste favorite** — étape C (bisection réseau) à faire |
| **H2** | Coin caché de mon SPLIT sous stalls (pas exclu : probes eager uniquement) | Revue de code propre mais preuve nulle | BMC identité « séquence sortie ≡ séquence entrée aplatie » (infra playbook existante) + sim randomisée valid/ready indépendants | ✅ **BLANCHIE** — 9/9 sessions sim bit-exactes + discipline vérifiée |
| **H3** | Interaction de **chaîne** 16→1→25 (deux RepackOps en série via lanes=1, ops/repack.scala:104-108) : chaque op sain isolément, couplage SPLIT→AGGREGATE suspect | Les ratios non multiples du vrai réseau passent TOUS par cette chaîne double | BMC sur la **chaîne complète** + probe dédié 16→1→25 avec stalls aléatoires | ✅ **BLANCHIE** — chaîne miniature 4→1→3 PROUVÉE en BMC(16) + probe sim 16→1→25 bit-exact |
| **H4** | Contrat partial-final-beat : un comptage suppose beats pleins quelque part (Tensor.scala:15-19 autorise le dernier beat partiel) | La cadence change la position relative du beat partiel | Assertion sim « total éléments reçus == shape.product » sur chaque branche du DAG | Non testée directement (les tailles du harnais A étaient divisibles) |

### M1.5 Plan de dissection proposé

Ordre croissant en coût, chaque étape donne un livrable exploitable même si le mystère résiste :

- **Étape A — Harnais différentiel à stalls aléatoires** (~½ journée)
  Géniteur : N éléments, `valid` aléatoire par cycle. DUT : op candidat (im2col, matmul,
  bias_add, Softmax1D, maxpool/avgpool 1d/2d, add, concatenate) précédé tantôt de l'adapter
  legacy, tantôt du RepackOp flushable. Checker : flatten logiciel bit-exact.
  **Livrable** : nom du/des ops coupables — ou blanchiment complet de H1, pointant H2/H3.
- **Étape B — Invariant BMC identité** (~½–1 j., infra `symbolicTestPlaybook.md` déjà en place)
  Propriété : ∀ séquence d'entrée avec ready aléatoire, sortie ≡ entrée aplatie. Sur RepackOp
  flushable SEUL (H2) puis sur la chaîne 16→1→25 (H3).
  **Livrable** : preuve, ou contre-exemple minimal + trace VCD.
- **Étape C — Bisection réseau** (quelques heures, une fois A/B outillés)
  Dans ResidualMLP : remplacer UN SEUL repack de graphe à la fois par sa variante legacy →
  isoler l'instance exacte qui fait diverger.
- **Étape D — Revue du consommateur localisé → fix → gates complètes**
  Chaîné+random combinés (rapport RC §5.3), DAG 6/6, suite 50/50, Python 3/3.

**Critère de clôture M1** : cause racine écrite ici + le harnais de l'étape A promu test de
non-régression permanent + décision assumée sur l'unification des gearboxes (suppression du
cloison carte §4.5 ou justification définitive du cloison).

### M1.6 RÉSULTATS DU SPRINT (2026-08) — étapes A et B exécutées

**Étape A — harnais différentiel à stalls aléatoires : ✅ FAIT, tout vert.**
`spinalML/test/src/spinalML/memory/RepackStallDiffTest.scala` (permanent, 9 tests) :
- valid/ready **indépendants et seedés**, 2 commandes consécutives par session (frontière +
  reArm + `isEmpty` vérifiés), golden absolu bit-exact sur chaque groupe émis
- Configs : SPLIT BF16 4→1 (la config exacte de ResidualMLP-image !) ×2 profils de stalls,
  SPLIT I8 4→1 et 16→4, AGGREGATE I8 2→4 ×2 profils, CHAÎNE I8 16→1→25 ×2 profils,
  cross-checks legacy
- **Résultat : 9/9 bit-exacts.** La gearbox structurée survive aux pires cadences randomisées.

**Étape B — invariants BMC identité : ✅ FAIT, deux preuves.**
- `RepackStallAggregateFormal` (BMC 14, CVC4) : handshakes anyseq libres ; preuves =
  exclusion mutuelle des fires (argument M1.2-① promu en théorème), jamais de groupe partiel
  exposé (`valid ⇒ pending==2`), identité de chaque émission avec la fenêtre glissante des
  beats acceptés.
- `RepackChainFormal` (BMC 16) : chaîne miniature non-multiple 4→1→3 (deux gearboxes via
  lanes=1, forme exacte du chemin poids W4A8 à échelle réduite) ; identité groupe-vs-flatten
  + garde « un groupe n'existe qu'après ses éléments » (LUT ceil).
- Piège d'API noté au passage : `subdivideIn(n slices)` = n morceaux (pas w bits !) ;
  la bonne forme est `subdivideIn(w bits)`.

**Leçon piège instrumentale (coûtée 3 itérations rouges)** : le `when(io.reArm)` du gearbox est
last-wins — un beat accepté AU cycle du pulse serait silencieusement perdu (capture puis flush).
Le vrai système ne peut jamais y arriver car `cmd.ready` est gated sur `gearboxEmpty`
(DMAReader.scala:71-73) : aucun beat ne précède sa frontière. Le harnais devait reproduire cette
discipline strictement (aucune émission entre détection de frontière et pulse). Toute future
utilisation directe d'un RepackOp flushable hors DMAReader devra respecter ce contrat.

**Bilan hypothèses après A+B** : H2 blanchi (sim), H3 blanchi (sim + formel), H1 reste favorite
et H4 non couverte. La gearbox elle-même est désormais **hors de cause** : si ResidualMLP
diverge quand on remplace son adapter legacy, le coupable est UN consommateur sensible au
nouveau pacing. → **Étape C (bisection réseau)** reste à faire pour nommer le coupable.

**Statut M1 : 🟡 EN COURS** — gearboxes blanchies et outillage permanent en place ;
prochaine action unique : étape C, quelques heures.

### M1.7 RÉSULTATS ÉTAPE C — BISECTION RÉSEAU (2026-08)

Reproduction sur demande obtenue avec la seule variable `BISECT_LINEAR_ONLY` (Linear-input
repack structuré), sortie historique exacte reproduite :

| # | Configuration | Résultat golden | Conclusion |
|---|---|---|---|
| C0 | Reader1D image structuré seul (axe A) | 🟢 | chemin image innocent (gate cmd.ready vide actif) |
| C1 | Linear#1 **ET** Linear#2 input-repack structurés | 🔴 EXACT `[…,0,0,.625,.25]` | reproduction complète |
| C2 | Linear#1 seul structuré | 🔴 EXACT | le coupable est DENSE1, pas Dense2 |
| C3 | Linear#2 seul structuré | 🟢 | confirmation C2 |
| C4 | + FIFO découplante 32 prof. **sortie** du site fautif | 🔴 identique | le pacing de SORTIE n'est pas en cause |
| C5 | TapBuffer profondeur ×2 | 🔴 identique | débordement FIFO hors de cause |
| C6 | Échange des deux vues du fork node0 | signature nouvelle `[1.5×8]` | sensible à la POSITION des consommateurs |
| C7 | + FIFO élastique profondeur 2/4/64 **entrée** du site OU `m2sPipe` entrée | 🟢 **GOLDEN EXACT** | le couplage `a.ready:=!full` ↔ amont partagé EST le mécanisme |
| C8 | Fix internalisé dans RepackOp (fifo/cut sur TOUS les flushables) | 🔴 autre signature (mot-frontière décalé entre couches) | la synchronisation inter-couches dépend de la géométrie des pacing ; un fix global change l'alignement matmul A/B |

**Cause racine (région prouvée)** : le mode flushable expose une backpression dure
(`a.ready := !full`, rafale-blocage). Attachée DIRECTEMENT à la branche directe d'un
fork partagé (Tee node0 → reshaped → repack 1→4), cette politique traverse toute la chaîne
de ready sans élasticité et perturbe l'état séquentiel partagé amont (double-buffer image /
streamer) au point que **l'AUTRE branche** du fork (FIFO différée du skip) livre une séquence
fausse (`[-1 dupliqué, -2 perdu]`) — d'où la corruption Add+ReLU observée, 100 % expliquée par
cette séquence d'entrée erronée (`ta+tb` élément-wise vérifié numériquement sur trace pré-edge).
Une simple élasticité d'un étage côté entrée restaure intégralement la correction (C7),
y compris pour `m2sPipe` profond-1 : la coupure doit être LOCALE AU POINT DE FIXATION,
pas interne au composant (C8 montre qu'un décalage global désynchronise la géométrie
A/B des matmuls en réseau).

**Leçon de fond (architecture)** : un consommateur à ready dur/combinatoire ne doit JAMAIS
pendre directement à une branche partagée d'un fork dont l'autre branche alimente de
l'état séquentiel sensible. La règle projet désormais : **tout attachement de gearbox
structurée sur un fan-out exige une étape élastique locale (FIFO ≥ 2 ou pipe paire)**.
Le cloison actuel image-legacy / poids-flushable est donc non seulement sûr empiriquement :
il est PRINCIPÉ — les readers poids/biais acceptent via `cmd.ready` gated (frontière de
commande, pas de couplage combinatoire durable), jamais directement sur un fan-out intra-graphe.

**Suivi optionnel restant** (micro-mécanisme RTL exact du mot-frontière perdu/dupliqué à
travers streamer/tee) : harnais standalone reproduisant topologie exacte + pacing matmul ;
ne bloque plus rien tant que la règle d'élasticité ci-dessus est appliquée aux nouveaux sites.

### M1.8 Pourquoi c'est bloquant AVANT la fin Phase 2 / début Phase 3

Le multi-tile fera circuler les images **en tuiles avec halos** : exactement des régions finissant
en milieu de groupe, traversant des consommateurs fortement stalling. C'est le régime précis où M1
vit. Concevoir le préfetch (Phase 2b) et le tiling (Phase 3) sans comprendre = construire deux fois.

---

## M2 — État de fenêtre persistant d'Im2ColOp

### M2.1 Les faits

`stateDone` remet les compteurs à zéro (ops/im2col.scala:206-215) mais **jamais** les trois
structures de fenêtre :

- `lineBuffers` — K−1 × W·C registres (im2col.scala:53-57)
- `shiftReg` — K·K·C registres (im2col.scala:60-61)
- `tempVecs` — K×C registres (im2col.scala:64-66)

Elles ont toutes un `init(zéro)` **power-on uniquement** (lignes 54-55, 61, 66) : les pixels de
l'image N survivent jusqu'à l'image N+1.

### M2.2 Preuve d'inofensivité actuelle (argument manuel, non formalisé)

1. Une fenêtre n'est émise que si `isWindowValid = (x ≥ K−1) && (y ≥ K−1)` (im2col.scala:83).
2. `x`/`y` repartent de zéro à chaque commande (`stateDone`, im2col.scala:206-215).
3. Donc toute fenêtre émise ne couvre que des colonnes/lignes **consommées pendant CETTE
   commande** : après K colonnes poussées, le shift register a expulsé tout contenu périmé vers
   la gauche.

⇒ Inoffensif tant qu'une commande va toujours jusqu'à son `stateDone`. **Limite** : c'est un
argument de papier ; aucun abort mid-commande n'existe aujourd'hui pour le contredire.

### M2.3 Risques par horizon

| Horizon | Exposition | Verdict |
|---|---|---|
| Résidence poids (Phase 2a) | Aucun direct : im2col consomme des **activations** (refluent à chaque inférence), pas les poids résidents | OK |
| Chaînage d'inférences | Couvert empiriquement : MnistChainedTest vert or Conv2D utilise im2col (layers/Conv2D.scala:28) | OK testé |
| Multi-tile halo (Phase 3) | Les line buffers deviennent l'endroit où VIVRE : l'état inter-tuiles implicite devra être remplacé par un fetch de halo **explicite** | Refonte prévue, pas un patch |
| Abort / soft-reset debug (rapport RC §5.4, phase 4) | Un reset mid-inférence laisserait les fenêtres dans un état arbitraire sans garde-fou | Deviendra bloquant CE jour-là |

### M2.4 Options et recommandation

| Option | Contenu | Coût | Quand |
|---|---|---|---|
| A | Vidage explicite (`io.reArm` port ou clear dans `stateDone`) des 3 structures | Non trivial : line buffers = W·C regs par ligne → clear large ou FSM multi-cycles | Assurance-vie si abort/soft-reset arrive |
| B | Preuve formelle de l'argument M2.2 (BMC : « aucune cellule périmée lue à l'émission ») | Coût > bénéfice aujourd'hui | Si M1 révèle des bugs de même famille |
| C | Statu quo documenté (ce fichier) + la preuve M2.2 comme commentaire de code | Déjà fait côté doc | ✅ **Recommandé maintenant** |

**Décision proposée** : Option C aujourd'hui ; Option A devient **obligatoire** le jour où un
mécanisme d'abort/soft-reset existe ; refonte halo-explicite en Phase 3 remplacera la question.

---

## Annexe A — Ce que SpinalHDL documentait DÉJÀ (à relire avant réemploi)

Le repo dupliqué `/home/leo/SpinalHDL-1.14.2` contient les avertissements qui décrivent nos
pièges — il faut prendre l'habitude de lire les commentaires des primitives AVANT de les câbler
(chemins relatifs : `lib/src/main/scala/spinal/lib/Stream.scala`) :

| Primitive | Ce que dit la lib | Ligne | Notre leçon |
|---|---|---|---|
| `StreamFork(synchronous=true)` | *« may lead to dead locks … also violates the handshake of the AXI specification (section A3.3.1) »* — car `outputs.valid := input.valid && input.ready` | Stream.scala:1328-1350 | Notre deadlock n°2 était écrit noir sur blanc. Règle carte §4.2 : fork async + frontière = front montant de `valid` |
| `StreamFork` (les DEUX modes) | *« The input stream will block until all output streams have processed each item regardlessly »* | Stream.scala:1325-1327 | `start.fire` est toujours une frontière trop tardive |
| `StreamWidthAdapter` up-conversion | Registre `buffer` + `Counter` **sans notion de commande** — rien ne se vide entre deux commandes | Stream.scala:2138-2152 | Cause racine RC2. Jamais de région finissant en milieu de groupe sans drain explicite |
| `StreamArbiter.LockPolicy.NoLock` | *« this may violate such handshake protocols »* (valid/payload doivent rester stables jusqu'au fire) | Stream.scala:836-841 | Pas utilisé directement aujourd'hui ; **À RELIRE avant l'arbitrage préfetch Phase 2b** |
| `StreamMux` / `StreamDemux` (sans joinSel) | *« Caution: the other direction is not synchronized »* | Stream.scala:1179,1220 | Pas utilisés ; note de lecture pour l'avenir |

**Habitude à instituer** : avant d'introduire une primitive lib sur un chemin de données,
`grep` ses commentaires dans le repo dupliqué — deux minutes qui valent des sessions de debug.

---

## Annexe B — Historique du registre

| Date | ID | Événement |
|---|---|---|
| 2026-08 | M1, M2 | Création du registre depuis carte §5 (zones d'ombre) et rapport RC §5 (follow-ups) |
| 2026-08 | — | Formels phase 1 (harnais avec ports `reArm`) : PASS en CI GitHub. Suivi follow-up §5.1 du rapport RC clos. |
| 2026-08 | — | Confirmation locale des mêmes suites (8/8 exit 0 : StreamDoubleBuffer, DMAReader, DMAReader2D, DoubleBufferStreamer, Conv1D, Conv2D, Linear, MatMul — CVC4 1.8). |
| 2026-08 | M1 | **Étape A** : harnais permanent `RepackStallDiffTest` (9 tests) — gearbox structurée blanchie sous stalls aléatoires multi-commandes, y compris config ResidualMLP-image BF16 4→1 et chaîne 16→1→25. |
| 2026-08 | M1 | **Étape B** : deux preuves BMC CVC4 — agrégat flushable 2→4 (`repack_flush_aggregate_stalls`, incl. théorème d'exclusion mutuelle) + chaîne non-multiple 4→1→3 (`repack_chain_4to1to3_stalls`). H2/H3 blanchies. |
| 2026-08 | — | Combo chaîné × aléatoire : `MNIST_CHAIN_SEED` ajouté à MnistChainedTest ; 20/20 bit-exact (N=10 × 2 modèles). |
| 2026-08 | M1 | **Étape C** : bisection réseau complète (C0–C8) — cause racine régionale prouvée (couplage ready-dur × fork partagé) ; règle d'architecture établie (élasticité obligatoire au fan-out) ; cloison actuel principé. Code expérimental reverté, `src/main` intact. |
