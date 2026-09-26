# Session 2026-09 — Hang du spill multi-beats : course reArm/fire du biais (S4)

## Symptôme

- `tests/universal/UniversalScaleDemo.scala` (Linear I8 K=1024, N=64, Ks=256,
  P=4, 64 KiB de poids) : timeout, 0/64 outputs, bus AXI idle.
- Bisect : spill 1-beat (M×N ≤ 8, ex. N=8) passe, spill ≥ 2 beats (N=16)
  bloque. Toutes les suites e2e existantes ne couvraient que du 1-beat en
  `Sequential` complet (`SequentialReplicaSpillTest` : M×N=4 → 1 beat) ; le
  2-beats ne passait qu'en wrapper isolé (`SpillPassLoopWrapper`, sans moteur).
- Le contrôleur sort de la prelude pass 1 (`pi=1`), tout le trafic AXI attendu
  est émis (AR=40 = W 32 + A 4 + bias 2 + seed 2), puis plus rien : `busy=1`,
  `passDone` jamais.

## Localisation (repro minimal + sondes)

Nouveau repro rapide `spinalML/test/.../nn/ScaleSpillDebugTest.scala`
(contrôle 1-beat N=8 + repro 2-beats N=16, timeout dur 20000 cycles, traces
dans `out/scale-debug*.log`, jamais d'attente interminable) + taps
`simPublic` comportement-neutres dans `Sequential` (seed/drain/A/W/bias/y +
`matmul→bias_add`, reArm) et ports debug `cMonV/cMonR` sur `LinearLayer`.

Comptes de fires côté moteur (pass 1 du repro) :
`seedF=16 drainF=16(pass0) aF=8 wF=128 biasF=16 yF=0 cF=0`,
`cV=1/cR=0`, `bV=0/bR=1`. Le matmul consomme tout (seed/A/W) puis cale en
`EmitRow` final (`c` valide, non accepté) ; `BiasAddOp` reste en `LoadBias`
(`b.ready=1`) avec son buffer vide.

Forensique VCD (`simWorkspace/Accelerator_1/test/wave.vcd`, run du repro) :
- `BiasAddOp.fsm` : `BOOT→LoadBias`, plus AUCUNE transition (jamais `Process`).
- `loadCounter` : 0→…→15 puis stop (15 fires, pas de 16ᵉ, pas d'overflow).
- `bias_reArm` : exactement 2 impulsions (START + prelude finale), pas de
  reArm parasite. AXI : 40/40 beats R reçus (aucune perte bus).
- Zéros (pass 0) : même signature — 15 consommés puis sec (voir § cause n°2).

## Cause racine (2 courses reArm/fire, même famille)

1. **`BiasAddOp.stateLoadBias` (fatale).** `when(reArm){loadCounter.clear()}`
   est évalué last-assignment-wins contre l'incrément du fire de la même
   cycle : un beat de biais parqué (`valid`) sur le cycle du reArm est
   consommé (`ready=1`) mais jamais compté. `loadCounter` finit à 15/16,
   jamais d'overflow → `LoadBias` pour toujours → `EmitRow` final calé →
   pas de `passDone` → hang contrôleur. Déclenché dès que le biais est
   parqué avant la prelude finale (cas typique à l'échelle : prelude 1
   retardée par le pacing seed 2-chunks pendant que le biais attend depuis
   longtemps ; à petite taille le biais arrive souvent après le reArm, d'où
   le « 1-beat passe » par chance de timing, pas par construction).
2. **`bMux` zero path (`LinearLayer`, idem `Conv1DLayer`/`Conv2DLayer`,
   bénigne ici).** `when(biasReArm){zeroRemain := N}` vs décrément sur fire,
   même conflit last-wins : 15 zéros au lieu de 16 (reset perdu). Sans effet
   sur la pass 0 (le moteur draine vers le spill, `c` sec), mais même
   fragilité.

## Fix (bulles élastiques d'un cycle, zéro changement fonctionnel)

- `ops/bias_add.scala`, `LoadBias` : `io.b.stream.ready := !io.reArm`
  (le beat reste parqué pendant le reArm, compteur repart propre, 16/16).
- `ops/bias_add.scala`, `Process` (défensif) : `a.ready` et `c.valid`
  gatés par `!io.reArm` (un fire A coïncidant avec l'abort de pass
  émettrait un `y` fantôme + désalignerait `aCounter`).
- `layers/Linear.scala`, `layers/Conv1D.scala`, `layers/Conv2D.scala`,
  mux zéro : `valid := zeroRemain =/= 0 && !biasReArm`.
- Aucun changement de protocole : en legacy comme en spill, le reArm ne
  tombe qu'à des frontières de commande (`c` sec), les bulles sont des
  no-ops hors du cas de course.

Fichiers : `spinalML/src/spinalML/ops/bias_add.scala`,
`spinalML/src/spinalML/layers/{Linear,Conv1D,Conv2D}.scala`
(+ taps debug `Sequential`/`LinearLayer`/`ScaleSpillDebugTest`, à nettoyer
ou garder pour le scale-up).

## Vérification

- `ScaleSpillDebugTest` 2/2 vert : contrôle 154 cycles, repro **478 cycles**
  (`yF=16 cF=16`, ex-timeout 20000).
- `spinalml test tests/universal/UniversalScaleDemo.scala` : **30350 cycles,
  bit-exact `dev=0.0`, 64/64 outputs** (`out/scale-universal.log`).
- `build tests/universal/UniversalScaleDemo.scala --dram --synth-only` :
  Yosys 57 s, **LUT 8873 (42,8 %), FF 5217, BSRAM 0, DSP 8**
  (`out/scale-build.log`, nextpnr non tenté — gap `_MEM` connu).

## Leçons bench (à réutiliser)

- Compter des FIRES cumulés, pas des niveaux échantillonnés : `aBeat`
  reboucle (0 après un sweep complet) et les impulsions tiennent 1 cycle —
  l'échantillonnage espacé seul avait conclu à tort « A ne coule pas ».
- Méfiance VCD : IDs réutilisés entre scopes (conflation de nets distincts),
  valeurs initiales dans `$dumpvars` (un `ready` haut dès t=0 fausse tout
  comptage de fires), `valid` haut N cycles = N beats (pas N transitions).
- Règle RTL : tout `when(reArm){counter.clear()}` cohabitant avec un fire
  du même cycle est suspect — discipliner par bulle (`ready/valid` gatés),
  comme les servants sticky du `SpillPassController`.
