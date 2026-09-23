# Session 2026-09 — Deadlock du seed spill conv P=2 (S2e)

## Où on en était

- Branche `ddrimpl3`, après merge PR #10 (réplica + stress).
- P0 (enveloppe Linear) clos et committé (P0a–P0d, gate 13 sim + 6 formels verts).
- P1 (spill Conv2D) : P1-1 → P1-4 committés (trait `SpillableGEMM`, plumbing
  fetch/contrôleur/sizing, moteur `Conv2DLayer` spill, layout slice-transposé +
  fold réplica `ConvSpillFoldTest` 4/4).
- P1-5 e2e (`ConvReplicaSpillTest`, driver partagé `ReplicaSpillE2E`) : dense
  inC=2 et P=1 bit-exacts, mais **P=2 I8 + BF16 : 0/50 outputs, timeout
  50000 cycles**. Deux bugs déjà fixés au passage (non committés à l'époque,
  committés depuis dans `76f0218`) :
  - plan image 2D ignorant les canaux (`strideBytesImg`, fetch court 36/72B) ;
  - ordre fenêtre réplica≠HW (`(c,r,k)` vs `(r,k,c)` du shiftReg im2col).

## Le gros souci : attente circulaire sur le bus in-order

Symptôme VCD (`simWorkspace/Accelerator/test/wave.vcd`, run P2) : passe 0
correcte (drain via `spillOut`, `passIdx` 0→1, seed AR 7 beats + refetch W +
re-fire image émis), puis **exactement 1 beat R transféré** (seed beat 1) et
`r_ready=0` pour toujours.

Mécanisme, prouvé par lecture du code (pas une hypothèse) :
1. Le contrôleur (`SpillPassController`) émettait le seed en **une seule
   commande région** (7 beats AXI pour 50 partiels I8, `spillBeats=7`).
2. `AxiMemorySim` sert en ordre : les 7 beats seed partent devant les beats
   image/W acceptés après.
3. Le moteur (`MatmulOp` S1 `stateSeedRow`, `matmul.scala`) ne consomme le
   seed que **ligne par ligne juste avant son LoadA** (fenêtre `temporal`,
   `accTable` slot-indexée `N*min(temporal,M)` — le seed eager est impossible,
   l'aliasing est réel, vérifié lignes 310-322).
4. Le gearbox flushable du seed reader ne tient qu'**1 beat** (`RepackOp`
   SPLIT `hold`+`full`, `repack.scala:38-61`). Le moteur prend ~1 beat puis
   attend la ligne A0, dont les beats image sont coincés derrière les 6 beats
   seed non consommables dans la mémoire in-order. Le seed attend le moteur
   qui attend A. **Boucle fermée, aucun timeout ne la résout.**
5. Linear ne la voyait jamais (seeds ≤ capacité gearbox).

Pistes écartées (avec raison) : plus gros buffer (tue le scaling spill),
compter sur du OoO mémoire (non garanti), reséquencement sans signal
(fragile dès `nBands>1`), chunks par ligne (les lignes du layout packé ne
sont pas alignées-beat → cascade layout/writer/réplica), seed eager
(aliasing, prouvé ci-dessus).

## Le fix S2e : seed en chunks d'1 beat pacés par consommation

Idée : borner le seed en vol à 1 beat. Un chunk est accepté **seulement
quand le gearbox est vide (reçu ET consommé)** = exactement le rythme de
consommation du moteur. Tout stall R se résorbe alors par consommation
moteur sans trafic bus → les beats image/W derrière finissent toujours par
passer. Preuve d'airtightness dans `SpillPassController.scala` (commentaire
S2e).

Changements (7 fichiers) :
- `nn/SpillPassController.scala` : boucle de beats (`beatIdx`, `length=0`,
  stride `beatBytes`, servant préambule + WaitPass), signal `readerFired`
  conservé (pin formel), nouveau param `beatBytes`.
- `nn/Sequential.scala` : seed readers conv+linear en `trimToElements=false`
  (le compteur trim restarte à chaque `cmd.fire`, incompatible avec le
  chunking) + flushable gardé (gate + phase), calcul du pad + requires
  d'égalité acc==lType, threading `spillPadElems`.
- `ops/matmul.scala` : `spillPadElems` + `stateDrainPad` (le pad de fin de
  région doit être drop-drainé avant `passDone`, sinon il bloque la gate du
  pass suivant ou empoisonne son seed row-0 ; 0 = pas de phase, zéro
  changement pour les régions beat-exactes) + threading `matmul.apply`.
- `layers/Linear.scala`, `layers/Conv2D.scala` : threading du param (un seul
  site consommateur S1 pour les deux chemins).
- `nn/SpillPassControllerTest.scala` : wrapper au standard S2e
  (`trim=false`), **nouveau test 2-beats** (M=2/N=8 : chunks base/base+8,
  loopback exact, preuve DDR), scripting converti à
  `StreamDriver.queue`/`StreamMonitor` (voir leçon bench).
- `symbolicTest/nn/SpillPassControllerFormal.scala` : prose du contrat
  (spillBeats chunks) + fence assouplie (pulse ou latch) ; propriétés
  inchangées (`spillBeats=1` au comportement identique).

Sous-bug trouvé par le test stub : `writerDone` mono-cycle arrivant **avant**
`passDone` (mémoire rapide + queue pad-drain moteur) — la fence le ratait.
Fixé par latch `writerDoneSeen` (discipline NN-01), côté DUT **et** côté
bench (sticky `wDoneSeen`). Sans le drain pad, `passDone` suit l'emit de
~0 cycle et le done est sauf ; avec, la fenêtre de course est réelle aussi
en prod — le latch n'est pas du luxe.

## Leçon bench (à réutiliser)

Le scripting manuel `ready/valid` (`toBoolean` + `tick`) est flaky dès qu'il
y a du backpressure : courses poke-then-edge avec les threads memSim,
patterns instables entre runs (beat perdu puis beat dupliqué, ou l'inverse).
Règle : **ne jamais échantillonner en thread test que des signaux dérivés
de registres horlogés** ; pour les handshakes, utiliser
`StreamDriver.queue`/`StreamMonitor` (`onSamplings`, synchronisé kernel).
Coût payé ici : 3 itérations de debug sur le seul test stub avant de
comprendre que le DUT était déjà correct.

## Vérification (état au moment d'écrire)

- `ConvReplicaSpillTest` **5/5** : P2 I8 (814c) + BF16 (766c) bit-exacts,
  AW=7/13 = exactement les region beats ; P1/DENSE/C1 inchangés.
- `SequentialReplicaSpillTest` 8/8 (K64-P8, M2, TAP, RERUN),
  `ConvSpillFoldTest`, `ReplicaSpillFoldTest`, `SpillConfigTest`,
  `MatmulSpillTest`, `SpillPassControllerTest` 3/3.

## Reste (plan §5 d'origine, phase "continuer")

1. Nettoyer `ConvReplicaSpillTest` (tests TMP + flags `debug`).
2. Relancer les formels contrôleur (`SpillPassControllerFormal`, `Liveness` —
   asserts retouchés) puis gate complet 13 sim + 6 formels.
3. Chaos conv (`DramChaosSpillTest` étendu ou équivalent) + `--stress` CLI.
4. `UniversalConvSpillDemo` CLI bit-exact.
5. Docs : statut P1 dans `docs/ddr_spill_ops.md`, checklist `ddr_impl.md`.
6. Commit : découper (fix S2e prod + tests + formel), message dans le style
   P0a–P1-4.
