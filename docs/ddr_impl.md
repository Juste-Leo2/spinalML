# DDR — Plan d'implémentation : tuyauterie générique d'abord, DDR ensuite

> **Statut** : plan de travail (ne pas implémenter la DDR avant les phases 1-4).
> **Principe** : préparer la tuyauterie avant la DDR.
> **Référence philosophie** : `docs/project_structure.md` §1 « 3-Layer Sandwich ».
> **Docs liés** : `docs/wave6_ddr_scaling_plan.md` (cap DRAM),
> `docs/pre_wave6_bug.md` (socle bugs DDR, tous corrigés),
> `docs/refactoring.md` (dette dual-target), `docs/roadmap_board.md` (matrice boards).

## 0. Règle d'or (issue de `project_structure.md`)

```
COUCHE 1 : CŒUR ALGORITHMIQUE (100% portable)
  Stream[Tensor], ops, lanes, Target — ZÉRO primitive vendor,
  ZÉRO adresse physique en dur.
        │ Interface neutre (AXI4)
        ▼
COUCHE 2 : ADAPTATEUR MÉMOIRE (MemoryAdapter)
  BramAdapter / SramAsicAdapter / DdrAdapter — seule frontière swappable.
        │ Broches & horloges
        ▼
COUCHE 3 : PHYSIQUE & BOARD (SoCTop / BoardTop, boards/*.json, *.cst/*.xdc)
  Contraintes pins, reset, UART, contrôleur DRAM, calibration.
```

Conséquence : **aucun `if Tang` / `if DDR3` / `if 0x10000` dans la couche 1**.
La DDR est une implémentation de la couche 2 + une fiche board de la couche 3.
Tout ce qui est fait avant la DDR doit marcher à l'identique sur
`BramAdapter` et `SramAsicAdapter`, en simulation seule.

## 1. État des lieux (point de départ réel, sept. 2026)

| Brique | Fichier | Constat |
|---|---|---|
| `Target` | `spinalML/src/spinalML/Target.scala:30-58` | Existe : `FPGA(family, useHardDsp)`, `ASIC(pdk, pipelineStages)`, `Simulation` + `fromString` / `current`. Utilisé par `HardwareMul`, `ArithmeticConfig`, `UartSoC`. |
| `MemoryAdapter.factory` | `spinalML/src/spinalML/memory/MemoryAdapter.scala:38-52` | Incomplète : `ASIC → SramAsicAdapter`, `FPGA/Simulation → BramAdapter`. **`DdrAdapter` absent** de la factory. Paramètres `imgBase=0x10000 / weightBase=0x20000` en dur. |
| `Accelerator` | `spinalML/src/spinalML/nn/Accelerator.scala:21-36` | Ne prend **ni `target` ni `MemoryAdapter`**, seulement `axiConfig` + knobs `tileHeight / temporal / inLanes`. Les CSR `0x08/0x0C/0x20` portent des adresses absolues sans allocateur. |
| `UartSoC` | `spinalML/src/spinalML/io/UartSoC.scala:21-33,81-86` | Partiellement générique : `target`, `memoryAdapterFactory: Option[...]`, `memoryWords/imgBase/weightBase`. Défaut `isAsic → SramAsicAdapter` sinon `BramAdapter` (POR `BOOT` éliminé en ASIC, lignes 42-57). |
| `DdrAdapter` | `spinalML/src/spinalML/memory/DdrAdapter.scala:15-24` | Pass-through vers contrôleur AXI externe. Stocke `imgBase/weightBase` mais **ne les utilise pas** (aucune translation, contrairement à `BramAdapter.scala:34-44`). Interlock global RAW/WAR + `wrReady` + `B.ID` post BUG-DDR-04/05/09 : correct pour bring-up, trop conservateur pour le recouvrement prefetch/compute. |
| `boards/` | `boards/tang-primer-20k.json`, `cli/spinalml_cli/board.py:104-133` | `limits: {lut, ff, bram, dsp}`, `bram_words: 4096`, `clk_freq`, `.cst` (clk/reset/UART uniquement). **Aucune section mémoire/DDR**. `board.py` ignore les clés inconnues → ajout sans casse possible, mais à exposer explicitement. |
| Layout | `spinalML/src/spinalML/utils/MemLayout.scala:17-24` | Source unique `regionBytes / alignToBeat`, déjà utilisée par `Sequential`. Pas d'allocateur `spill/out`, pas de `reportMemoryFit()` à l'élaboration. |

## 2. Phase 1 — Généralisation `Target` (tuyauterie, 1-2 j, sans hardware)

Objectif : le même `modelSpec` élabore pour sim / FPGA / ASIC sans changer une ligne.

1. **Propager `target: Target` dans `Accelerator` puis `Sequential`.**
   - Défaut `Target.Simulation` pour ne rien casser en CI.
   - `Accelerator` transmet à `Sequential`, aux couches `HardwareMul`/`ArithmeticConfig`, et au choix `DMAWriter` (pas de changement fonctionnel à ce stade).
   - `UartSoC.acceleratorFactory` construit l'accélérateur avec le même `target` que le SoC (aujourd'hui les deux sont indépendants).
2. **Compléter `MemoryAdapter.factory`.**
   - Ajouter le cas DDR : `case (FPGA, DDR) → DdrAdapter`, `case (ASIC, …)` inchangé, `Simulation → BramAdapter` par défaut.
   - Introduire un sélecteur explicite `MemoryKind = OnChip | ExternalDram | AsicSram` plutôt que de deviner depuis `Target` seul (un FPGA peut vouloir BRAM en test et DDR en prod).
   - Garder la signature compatible (défauts identiques) pour ne pas casser `MemoryAdapterTest.scala:19-25` ni `UartSoCTest`.
3. **Câbler `Target.current` / `SPINALML_TARGET` / `--no-dsp` jusqu'à la CLI.**
   - `Target.fromString` existe déjà ; l'exposer dans `spinalml compile/build/test` et logguer la cible résolue dans chaque run.
4. **Gate Phase 1.**
   - `MemoryAdapterTest`, `UartSoCTest` (FPGA + ASIC), `AcceleratorTest` verts.
   - `SpinalConfig.generateVerilog` bit-identique à `target` égal, Verilog ASIC sans blackbox Gowin (`DspMulTest` pattern).
   - Interdit : aucune adresse physique nouvelle, aucune pin DDR.

## 3. Phase 2 — `MemoryMap` générique (tuyauterie, 2-3 j, sans hardware)

Objectif : fini les `0x10000/0x20000` dispersés (`UartSoC:27-28`, `AxiReadMem`, `cli.py`, `Sequential.currentMemoryOffset`, `Accelerator.outBaseOffset`).

1. **Créer `nn/MemorySpec.scala` (couche 2, logique, pas physique).**
   ```scala
   case class MemoryRegion(base: Long, sizeBytes: Long)
   case class MemorySpec(
     kind: MemoryKind,          // OnChip | ExternalDram | AsicSram
     img: MemoryRegion,
     weights: MemoryRegion,
     spill: Option[MemoryRegion],  // None = pas de spill (modèles on-chip)
     out: MemoryRegion
   )
   ```
   - `Accelerator(memory: MemorySpec)` remplace les `imgBase/weightBase: Int` épars.
   - `MemLayout.regionBytes / alignToBeat` reste la seule formule d'alignement beat.
2. **Allocateur partagé + `reportMemoryFit()`.**
   - Remplace le compteur local `Sequential` et `outBaseOffset` local `Accelerator` par un allocateur unique `img → weights (beat-aligné par région) → spill → out`.
   - À l'élaboration : `require` clair si empreinte > capacité déclarée (voir Phase 3), avec conseil (`active spill / tileHeight / folding`, cf. Wave 6 P1-P4).
   - Prévoit les curseurs génériques `frameCursor` (image) + `spillCursor` + `outCursor` (généralisation des `imgBaseOffset/outBaseOffset` actuels), reset sur write CSR de base.
3. **Figer la CSR (sans l'étendre au-delà du besoin).**
   - Map existante inchangée : `0x00 START, 0x04 STATUS, 0x08 IMG_BASE, 0x0C WEIGHT_BASE, 0x10 MODE, 0x14 RELOAD, 0x1C RUN, 0x18 TILE_CNT, 0x20 OUT_ADDR, 0x24 OUT_CTRL, 0x28 DMA_STATUS, 0x30 DEQUANT_SCALE`.
   - Réserve `0x34 SPILL_BASE` + `0x38 OUT_STRIDE` (au lieu du `outBytesAcc` interne) + `0x3C MEM_STATUS`, même si non câblés avant la Phase 5. Exporte la map en header Python/C (dette `full_roadmap.md` §9).
4. **Gate Phase 2.**
   - Même modèle compile en `OnChip` et `ExternalDram` simulé sans changer `modelSpec`, seules les bases changent.
   - Tests `AcceleratorTest` (résidence/prefetch/bandes/continu) verts dans les deux configs.
   - Interdit : aucun accès à un contrôleur réel, aucune `.cst` modifiée.

## 4. Phase 3 — `boards/*.json` + `board.py` (tuyauterie couche 3, 1 j, sans RTL)

Objectif : la capacité DDR devient une donnée board vérifiable avant synthèse.

1. **Ajouter un bloc `memory` au JSON (exemple pour Tang Primer 20K, valeurs à confirmer sur schéma) :**
   ```json
   "memory": {
     "onchip_words": 4096,
     "onchip_addr": {"imgBase": "0x10000", "weightBase": "0x20000"},
     "ddr": {"present": false, "size_bytes": 0, "base": "0x00000000",
             "dataWidth": 16, "controller": "none", "max_burst": 256}
   }
   ```
   - `present: false` tant que le contrôleur n'est pas calibré (le `.cst` actuel n'a que clk/reset/UART). Ne pas promettre une DDR3 qui n'est ni pinnée ni validée.
   - Quand le bring-up arrivera : `present: true, size_bytes: 67108864, controller: "gowin-ddr3-ip", …`.
2. **Exposer dans `cli/spinalml_cli/board.py:104-133`.**
   - Parser `raw_memory`, défauts `present=False`, `size_bytes=0`, `onchip_words=bram_words` (compat ascendante).
   - Utiliser pour : (a) nourrir `MemorySpec` par défaut du build, (b) garde-fou `empreinte > size_bytes → erreur claire`, (c) smoke synthèse `test-all` (cf. Wave 6 P0 : BSRAM > 46, `using FF mapping`, Fmax).
3. **Documenter `roadmap_board.md` : colonne `Mémoire (on-chip / DDR)` par board.**
4. **Gate Phase 3.**
   - `load_board_config("tang-primer-20k")` expose `memory.ddr.present=False` sans casser les builds existants.
   - `spinalml build --board tang-primer-20k --synth-only` échoue proprement si le modèle dépasse `onchip_words` sans spill.

## 5. Phase 4 — Étapes intermédiaires génériques (2-4 j, toujours en sim)

Objectif : scaling sans DDR physique. Tout est validé sur `BramAdapter` / `AxiMemorySim`.

1. **Fencing par région (remplace le stall global).**
   - `DdrAdapter` actuel bloque tout `AR` pendant tout write et tout `AW` pendant toute lecture. Extraire `FenceConfig(strictDram=true, relaxedSram=false)` : `Bram/SramAsic` sans stall, `Ddr` strict par défaut. Prépare le recouvrement prefetch/compute + spill Read-Modify-Write (Wave 6 P1).
2. **Arbiter cascade + lanes uniformes (Wave 6 P3, version générique).**
   - Remplace la soustraction `idWidth` (`Sequential` BUG-DDR-06, pansement `idWidth=8`) par un arbre `Axi4ReadOnlyArbiter` à 2 étages.
   - Étend `Linear.weightLanes` à `Conv2D/Conv1D/Attention` via un knob unique `maxWeightLanes/streamingWidth`. Préserve le contrat beat partiel (`Tensor` : `totalElements % lanes != 0` autorisé).
3. **Spill générique (Wave 6 P1, version `MemoryAdapter`, pas `DDR`).**
    - **Mémoire (FAIT)** : CSR `0x34 SPILL_BASE` live dans `Accelerator`
      (init depuis `MemorySpec.spillBase`, R/W hôte), `spillBytes` dans le
      fit check, `CsrMap`/`MemorySpecFormal` à jour (13 wired / 2 reserved).
    - **Compute (FAIT — S0/S1/S2 validées, 20/09/2026 ; reste S3 formel)** : K-split du GEMM
      avec sommes partielles M×N en DDR entre passes. Soudure identifiée :
      le drain temporal existant (`MatmulOp` `stateEmitRow`/`stateOutput`,
      drain + clear par ligne) devient la sortie spill sur passes non
      finales (mux `spillOut`/`io.c`) ; l'accumulation est ensemencée depuis
      `spillIn` sur passes > 0 (mux zéro/spill, sommes pleine largeur +
      bias final unique — contrainte `docs/ddr_replica_status.md` §2) ;
      boucle de chunks bornée à la slice + compteur de passes ; tranches W
      par offset (`reqW.address`), re-stream A depuis source DDR-residente
      (couche 1 d'abord) ; bias-zéro sur passes non finales (mux de stream,
      `BiasAddOp` consomme exactement N beats/commande) ; paire
      `DMAWriter`/`DMAReader` spill dans `Sequential` + curseur spill (reset
      sur write `0x34`) ; fold `spillWidth` côté réplica ; spec formelle du
      contrôleur multi-passes (progression, bias unique, pas de deadlock).
      Périmètre v1 : `Linear` à A DDR-résident (cas MLP-4k : la première
      couche est la grosse) ; `Conv`/couches profondes ensuite.
    - **Contrat runtime (S2)** : `STREAM_PER_PASS` seul (`CSR 0x10 = 0`) —
      sous residency le contrôleur supprime `refetchW` (stall bruyant, pas de
      corruption silencieuse) ; `temporal >= 1` ; une seule couche spillée
      (v1) ; re-stream A = re-fire image DDR (nœud 0 exclusif) ou replay
      `StreamTap` budgeté (nœud profond/partagé).
    - **Fencing inter-passes** : le prelude p+1 n'est émis qu'après le B du
      drain p (`writerDone`) — RAW sur la région spill unique. La visibilité
      inter-masters dépend de l'ordre du contrôleur mémoire (fence strict
      `DdrAdapter`, item 1) ; séquentiellement cohérente sur `AxiMemorySim`.
    - **Preuves e2e** : `SequentialSpillTest` 10/10 (P=2/P=1/zero-tail/tap
      profond/échelles K8-K64/rerun `TILE_CNT`-STOP-`MODE`-`0x34`) ; écarts et
      contrat layout W slice-transposé consignés dans
      `docs/ddr_final_impl.md` §S2.
    - Testé sur `AxiMemorySim` spill (oracle scala) ; la comparaison
      `ModelReplica` complète, le formel contrôleur et la non-régression
      totale restent en S3.
4. **Gate Phase 4.**
   - `Linear 4096 / Conv 7x7` élabore dans le budget (`maxWeightLanes` + `temporal`), spill bit-exact, formels `DMAWriterFormal` / `AcceleratorFormal` verts.
   - Interdit : toujours aucune dépendance au silicium Tang.

## 6. Phase 5 — Implémentation DDR (seulement après 1-4)

1. **P5a — `DdrAdapter` comme implémentation comme les autres (sim, 1-2 j).**
   - Même IO `MemoryAdapter` (`axi / wrEnable / wrAddr / wrData / wrStrb / wrReady`), même `B.ID`, fencing de Phase 4.
   - E2E sim : top à poids + `DdrAdapter` + `AxiMemorySim`, motifs AR (résidence = 0 AR steady, prefetch eager = 0 AR dans fenêtre START→premier beat), write-back vérifié mot-à-mot (`tests/python/test_accelerator.py` pattern).
2. **P5b — Bring-up Tang Primer 20K (hardware, 3-5 j + silicium).**
   - `.cst` DDR (pins, horloge), contraintes timing, calibration contrôleur Gowin.
   - Bascule `boards/tang-primer-20k.json : memory.ddr.present=true` + `size_bytes` réelle.
   - Lancer un modèle **strictement plus gros que la BRAM** en flux continu, poids résidents, bande passante/latence/LUT/FF/BSRAM/Fmax consignées (métrique Wave 6 §0).
   - Runner hôte Radxa optionnel.
3. **Gate Phase 5.**
   - Bit-exact sim + silicium sur le même `modelSpec`, `TILE_CNT` cohérent, STOP propre, `0x08` readback = base hôte (curseur interne).
   - Aucun `if Tang` en couches 1-2 ; tout le spécifique vit en couche 3 (`boards/`, `.cst`, `build_runner.py`, `flash_runner.py`).

## 7. Ordre, dépendances, non-objectifs

| Ordre | Chantier | Dépend de | Durée |
|---|---|---|---|
| 1 | `Target` propagé + factory complète | — | 1-2 j |
| 2 | `MemorySpec/MemoryMap` + CSR figée | 1 | 2-3 j |
| 3 | `boards/*.json` `memory/ddr` + `board.py` | 2 | 1 j |
| 4 | Fencing région + arbiter cascade + lanes + spill générique | 1-3 | 2-4 j |
| 5a | `DdrAdapter` e2e en sim | 1-4 | 1-2 j |
| 5b | Bring-up DDR3 Tang + mesures | 5a | 3-5 j + HW |

**Non-objectifs de ce document** (Wave 6 P2/P4, roadmap) : DAG tap multi-tile, folding L1/L2, ONNX importer, UART multi-octets. Ils se branchent sur la tuyauterie des phases 1-4 sans la modifier.

**Garde-fous** : chaque phase arrive avec son e2e en CI (`MemoryAdapterTest`, `AcceleratorTest`, `MLAcceleratorTest`, formels BMC, smoke `--synth-only`), sinon on reconstitue la dette P0 (suites archivées hors CI).
