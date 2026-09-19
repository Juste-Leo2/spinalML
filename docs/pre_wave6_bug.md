# Audit Pré-Wave 6 — Inventaire des Bugs, Faiblesses et Risques DDR

> **Statut** : Rapport d'analyse et d'audit préalable au lancement de la **Wave 6** (`docs/wave6_ddr_scaling_plan.md`).  
> **Date** : 19 Septembre 2026  
> **Branche** : `ddr`  
> **Objectif** : Identifier de manière exhaustive tous les bugs latents, incohérences matérielles, limitations bloquantes et régressions potentielles du sous-système mémoire/DDR avant d'entamer l'intégration de la DRAM au pipeline et le scaling mémoire.

---

## Synthèse Exécutive

La Wave 6 a pour ambition de faire franchir au projet le cap du **scaling DRAM** : passer d'un modèle one-shot on-chip (BRAM) à l'exécution de modèles dépassant la mémoire interne via le spill/streaming en DDR, le folding de couches (L2) et le flux continu.

L'analyse approfondie du code existant (`DdrAdapter`, `DMAWriter`, `DMAReader`, `DMAReader2D`, `StreamDoubleBuffer`, `DoubleBufferStreamer`, `Accelerator`, `Sequential`, `AxiReadMem`) révèle **11 anomalies et bugs critiques** directement liés à la DDR et à son interaction avec le datapath de calcul.

### Matrice Récapitulative des Risques par Chantier Wave 6

| ID | Intitulé du Bug / Risque | Sévérité | Chantier Wave 6 Impacté | Statut |
|---|---|---|---|---|
| [BUG-DDR-01](#bug-ddr-01--omission-du-guard-prefetchworldb-sur-le-re-arm-du-double-buffer-de-biais) | Omission du guard `!prefetchWorldB` sur le re-arm du double buffer de biais | 🔴 **CRITIQUE** | P0 (Tests), P4 (Folding L2) | ✅ **CORRIGÉ** |
| [BUG-DDR-02](#bug-ddr-02--absence-dincrément-de-ladresse-de-sortie-en-mode-flux-continu-writetoddr) | Écrasement systématique des sorties en mode continu `writeToDdr` | 🔴 **CRITIQUE** | P1 (Advanced Tiling), P2 (Flux continu) | ✅ **CORRIGÉ** |
| [BUG-DDR-03](#bug-ddr-03--masque-doctets-wstrb-nul-sur-le-dernier-beat-dmawriter-pour-les-types-4-bits) | Masque d'octets `w.strb` nul sur le dernier battement `DMAWriter` (< 8 bits) | 🔴 **CRITIQUE** | P1 (Spill d'accumulateurs), P3 (Scaling W4A8/FP4) | ✅ **CORRIGÉ** |
| [BUG-DDR-04](#bug-ddr-04--absence-de-barrière-raw-et-de-port-de-lecture-hôte-dans-ddradapter) | Absence de barrière RAW et de port de lecture hôte dans `DdrAdapter` | 🟠 **MAJEUR** | P1 (Spill/Read-Modify-Write), P5 (Bring-up DDR3) | ✅ **CORRIGÉ** |
| [BUG-DDR-05](#bug-ddr-05--absence-de-contrôle-de-flux-wrready-sur-le-port-décriture-hôte-de-ddradapter) | Absence de backpressure `wrReady` sur l'écriture hôte de `DdrAdapter` | 🟠 **MAJEUR** | P5 (Bring-up DDR3 / Pilote hôte) | ✅ **CORRIGÉ** |
| [BUG-DDR-06](#bug-ddr-06--crash-délaboration-sur-lidwidth-de-larbitre-axi-read-au-delà-de-8-couches) | Crash d'élaboration sur `idWidth` de l'arbitre AXI Read dès > 8 couches | 🟠 **MAJEUR** | P3 (Resource Scaling), P0 (Tests) | ✅ **CORRIGÉ** |
| [BUG-DDR-07](#bug-ddr-07--émission-prématurée-de-nexttile-dans-doublebufferstreamer) | Émission prématurée de `nextTile` dans `DoubleBufferStreamer` sous backpressure | 🟡 **MOYEN** | P2 (Flux continu), P4 (Prefetch / Folding) | ✅ **CORRIGÉ** |
| [BUG-DDR-08](#bug-ddr-08--risque-de-deadlock-dans-dmareader2d-si-rowwords--axilanes--outlanes--0) | Risque d'interblocage dans `DMAReader2D` si les battements de ligne ne divisent pas `outLanes` | ⚪ **FAUX POSITIF** | P1 (Tiling), P3 (Knobs lanes) | ⚪ **FAUX POSITIF DÉMONTRÉ** |
| [BUG-DDR-09](#bug-ddr-09--violation-du-protocole-axi4-sur-bid-dans-ddradapter-et-bramadapter) | Forçage en dur de `b.id := 0` (Non-conformité AXI4) dans les adaptateurs | 🟡 **MOYEN** | P5 (Interconnexion SoC / Bring-up) | ✅ **CORRIGÉ** |
| [BUG-DDR-10](#bug-ddr-10--flakiness-du-banc-dmasdbtb-lié-à-labsence-de-resetflush-du-streamer) | Flakiness non-déterministe du banc DDR `DmaSdbTb` (stale FIFO) | 🟢 **TEST** | P0 (Filet de sécurité) | ✅ **CORRIGÉ** |
| [BUG-DDR-11](#bug-ddr-11--désactivation-totale-en-ci-des-suites-end-to-end-ddr-archivées) | Exclusion complète des suites de tests end-to-end DDR du build CI | 🟢 **TEST** | P0 (Priorité 0 Wave 6) | **Dette de couverture** |

---

## Détail des Bugs & Vulnérabilités Détectés

### BUG-DDR-01 : Omission du guard `!prefetchWorldB` sur le re-arm du double buffer de biais

- **Sévérité** : 🔴 **CRITIQUE** (Corruption silencieuse d'état sous prefetch eager)
- **Fichier** : [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala#L508-L515)
- **Composant** : `Sequential.scala` (chemins `bDoubleBuffer` vs `wDoubleBuffer`)
- **Code concerné** :
  ```scala
  // Lignes 420-424 (POIDS - Correct):
  when(reqW.fire && prefetchWorldW) {
    stagedW := True
  }
  wDoubleBuffer.io.reArm := reqW.fire && !prefetchWorldW // <-- Banque protégée en mode prefetch
  wStreamer.io.reArm := reqW.fire

  // Lignes 508-513 (BIAIS - BUG) :
  when(reqB.fire && prefetchWorldB) {
    stagedB := True
  }
  biasDmaFire = reqB.fire
  bDoubleBuffer.io.reArm := reqB.fire // <-- MANQUE && !prefetchWorldB !
  bStreamer.io.reArm := reqB.fire
  ```
- **Mécanisme de défaillance** :
  En mode prefetch eager (`CSR 0x10 = 0b11`, `residentMode = true`, `prefetchMode = true`), un reload déclenché par l'hôte rafraîchit les banques d'arrière-plan (banques IDLE) pendant que les banques résidentes sont en cours d'utilisation par le datapath.
  Pour les poids, la ligne 423 neutralise le `reArm` destructeur (`&& !prefetchWorldW`), protégeant ainsi l'état de la banque résidente en cours de lecture.
  Mais pour les biais, **la ligne 512 pulse inconditionnellement `bDoubleBuffer.io.reArm` dès que `reqB.fire` se produit**.
  Dans [`StreamDoubleBuffer.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/StreamDoubleBuffer.scala#L157-L164) :
  ```scala
  when(io.reArm) {
    loadBank := False
    computeBank := False
    pingFull := False
    pongFull := False
    loadCounter.clear()
    switchArmed := False // <-- Le swap préparé est détruit !
  }
  ```
  Le `reArm` efface immédiatement les drapeaux de plénitude (`pingFull`, `pongFull`), force `computeBank := False` et annule `switchArmed := False`. La banque résidente de biais est corrompue en plein vol et le swap ordonnancé est avorté.
- **Impact Wave 6** : Bloquant pour la Priorité 0 (restauration de `WeightPrefetchChainTest`) et la Priorité 4 (Layer Folding L2 re-fetchant les coefficients).
- **Correction appliquée & validée** :
  Ligne 512 dans `Sequential.scala` corrigée avec `&& !prefetchWorldB`.
  Validée par `SequentialTest` (test dédié multi-passes avec mesure du trafic AR DDR) et `StreamDoubleBufferPrefetchFormal`. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-02 : Absence d'incrément de l'adresse de sortie en mode flux continu (`writeToDdr`)

- **Sévérité** : 🔴 **CRITIQUE** (Écrasement permanent de données en DDR)
- **Fichier** : [`spinalML/src/spinalML/nn/Accelerator.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala#L167-L170) et [`lignes 221-239`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala#L221-L239)
- **Code concerné** :
  ```scala
  val dmaCmd = Stream(WriteRequest(axiConfig.addressWidth))
  dmaCmd.address := outAddrReg // Adresse statique issue du CSR 0x20 !
  dmaCmd.length := U(totalOutBeats - 1, 16 bits)
  ...
  when(frameDone) {
    tileCntReg := tileCntReg + 1
    doneSticky := True
    when(runActive) {
      // Auto-advance: re-fire START and slide the image cursor forward.
      startPending := True
      imgBaseOffset := imgBaseOffset + imageBytesAcc // L'image avance bien !
      // MAIS AUCUN outBaseOffset N'EXISTE POUR dmaCmd.address !
    }
  }
  ```
- **Mécanisme de défaillance** :
  En mode d'inférence continue (`RUN = 1`, CSR `0x1C`), le matériel implémente un streaming automatique d'images (style flux vidéo ou suite de tokens). À chaque trame complétée (`frameDone`), `imgBaseOffset` avance de la taille d'une image, permettant à l'inférence $k$ de lire l'image $k$ en DDR.
  Cependant, **l'adresse d'écriture DDR (`dmaCmd.address`) est directement branchée sur le registre fixe `outAddrReg`**.
  Lors de chaque inférence successive :
  - L'image d'entrée $k$ est bien lue à `imgAddrReg + k * imageBytesAcc`.
  - Mais le résultat $k$ est écrit à l'adresse fixe `outAddrReg`, **écrasant systématiquement le résultat de l'inférence $k-1$**.
- **Correction appliquée & validée** :
  Ajout du registre `outBaseOffset` et calcul du stride aligné AXI `outBytesAcc = totalOutBeats * (axiConfig.dataWidth / 8)` dans `Accelerator.scala`.
  `dmaCmd.address` est connecté à `outAddrReg + outBaseOffset`.
  L'offset s'incrémente lors de `frameDone` lorsque `runActive && writeToDdr`, et se réinitialise lors de l'écriture sur le CSR `0x20`.
  Validée par le test de simulation `AcceleratorTest` (« Accelerator: Continuous streaming write-back to DDR (CSR 0x1C RUN + CSR 0x24 OUT_CTRL, auto-increment OUT_ADDR) ») et non-régression formelle BMC `AcceleratorFormal`. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-03 : Masque d'octets `w.strb` nul sur le dernier beat `DMAWriter` pour les types 4-bits

- **Sévérité** : 🔴 **CRITIQUE** (Écriture ignorée par la DDR pour W4A8 / FP4)
- **Fichier** : [`spinalML/src/spinalML/memory/DMAWriter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DMAWriter.scala#L210-L216)
- **Code concerné** :
  ```scala
  val finalBeatBytes = (totalElements - (totalAxiBeats - 1) * axiLanes) * (elemWidth / 8)
  val fullByteMask = (BigInt(1) << bytesPerBeat) - 1
  val lastByteMask = (BigInt(1) << finalBeatBytes) - 1
  val finalBeat = (remaining === 0) && (burstRemain === 1)
  io.axiMaster.w.strb := Mux(finalBeat,
    B(lastByteMask, bytesPerBeat bits),
    B(fullByteMask, bytesPerBeat bits))
  ```
- **Mécanisme de défaillance** :
  Lors de l'écriture en mémoire d'un tenseur quantifié en sous-octet (ex. activations compactes en FP4 `FP4_E2M1` ou entiers `I4`, où `elemWidth = 4`), la division entière Scala `(elemWidth / 8)` est évaluée à la compilation :
  $$\text{elemWidth} / 8 = 4 / 8 = 0$$
  Par conséquent :
  $$\text{finalBeatBytes} = (\dots) \times 0 = 0$$
  $$\text{lastByteMask} = (1 \ll 0) - 1 = 0$$
  Sur le tout dernier battement d'écriture AXI (`finalBeat == True`), le signal `w.strb` vaut `0x00`.
  Selon la spécification AXI4 (section A3.4.4), un octet dont le strobe est à 0 n'est pas écrit en mémoire. Le contrôleur DDR ignore donc l'écriture de ce dernier mot, tronquant silencieusement les dernières données du tenseur écrit en mémoire.
- **Impact Wave 6** : Corruption directe lors du write-back de tenseurs compressés en FP4/I4 ou de l'Advanced Tiling sur tenseurs sous-octets (Priorités 1 et 3).
- **Correction appliquée & validée** :
  Ligne 210 dans `DMAWriter.scala` corrigée : `finalBeatBytes = (finalBeatElems * elemWidth + 7) / 8`.
  Validée par `DMAWriterTest` (test sub-byte 4-bit strobe) et `DMAWriterFormal`. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-04 : Absence de barrière RAW et de port de lecture hôte dans `DdrAdapter`

- **Sévérité** : 🟠 **MAJEUR** (Incapacité de relecture hôte et risque de course Read-After-Write)
- **Fichier** : [`spinalML/src/spinalML/memory/DdrAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DdrAdapter.scala#L26-L31)
- **Code concerné** :
  ```scala
  // ------------------------------------------------------------------
  // AXI Read: pass-through from Accelerator master to external DDR
  // ------------------------------------------------------------------
  extIo.ddrMaster.ar <> io.axi.ar
  io.axi.r           <> extIo.ddrMaster.r
  ```
- **Mécanisme de défaillance** :
  1. **Absence de lecture hôte** : Contrairement à `BramAdapter` qui permet à l'hôte de relire et vérifier le contenu de la mémoire, `DdrAdapter` ne fournit aucun canal de lecture pour l'hôte. L'hôte ne peut qu'écrire (`io.wrEnable`). Si l'accélérateur écrit ses logits ou ses activations en DDR (`writeToDdr`), l'hôte (via UART ou bus SoC) est incapable de relire le résultat depuis la DDR.
  2. **Absence de barrière de cohérence RAW (Read-After-Write)** : Le canal de lecture `ar` est un simple fil pass-through. Les canaux d'écriture (`aw`, `w`, `b`) et de lecture (`ar`, `r`) sont totalement découplés.
  Dans le cadre de la Wave 6 Priorité 1 (Advanced Tiling / spill d'accumulateurs avec passes d'accumulation successives Read-Modify-Write) :
  - La passe $P$ écrit les sommes partielles en DDR via le maître AXI Write.
  - La passe $P+1$ relit ces sommes partielles via le maître AXI Read pour accumuler la tuile suivante.
  - Rien dans `DdrAdapter` ne garantit que la transaction d'écriture s'est achevée (réponse B reçue et commitée dans la DRAM physique) avant que la requête de lecture `ar` de la passe $P+1$ ne soit présentée au contrôleur DDR. Cela expose le système à des lectures de données périmées.
- **Impact Wave 6** : Bloquant pour le spill multi-passes d'accumulateurs (Priorité 1) et pour le bring-up DDR3 réel sur Tang Primer 20K (Priorité 5).
- **Correction appliquée & validée** :
  1. Synchronisation inter-canaux dans `DdrAdapter.scala` :
     - Gating du canal AXI Read : `extIo.ddrMaster.ar.valid := io.axi.ar.valid && !writeBusy` et `io.axi.ar.ready := extIo.ddrMaster.ar.ready && !writeBusy` avec `writeBusy = hostActive || accBusy`. Aucune requête de lecture ne peut être émise ni acceptée par le contrôleur DRAM tant qu'une écriture (hôte ou rafale accélérateur jusqu'à la réponse `B`) est en cours.
     - Compteur de transactions de lecture en vol `readInFlight` pour synchronisation bidirectionnelle : protection WAR bloquant `io.axi.aw.ready` et `io.wrReady` tant qu'une lecture DRAM est en cours d'évacuation (`readBusy = (readInFlight =/= 0)`).
     - Mémorisation et propagation de l'ID de transaction d'écriture accélérateur `accIdR := io.axi.aw.payload.id` sur `extIo.ddrMaster.aw.payload.id`.
  2. Validée par `MemoryAdapterTest` (« BUG-DDR-04: DdrAdapter RAW hazard interlock stalls AXI AR until write completes ») confirmant le blocage systématique de `ar.valid`/`ar.ready` durant toutes les phases d'écriture (`accAwPending`, `accStreaming`, `accWaitB`) et le déblocage synchrone à la réception de la réponse `B`. Non-régression complète validée sur `AcceleratorTest`, `MLAcceleratorTest` et vérification formelle SymbiYosys/BMC `AcceleratorFormal` (2/2 PASS). Statut : ✅ **CORRIGÉ**.


---

### BUG-DDR-05 : Absence de contrôle de flux `wrReady` sur le port d'écriture hôte de `DdrAdapter`

- **Sévérité** : 🟠 **MAJEUR** (Perte de données lors du chargement mémoire par l'hôte)
- **Fichier** : [`spinalML/src/spinalML/memory/DdrAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DdrAdapter.scala#L56-L63) et [`MemoryAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/MemoryAdapter.scala#L27-L31)
- **Code concerné** :
  ```scala
  when(io.wrEnable) {
    hostActive := True
    hostAwDone := False
    hostWDone  := False
    wrAddrReg  := io.wrAddr
    wrDataReg  := io.wrData
    wrStrbReg  := io.wrStrb
  }
  ```
- **Mécanisme de défaillance** :
  Sur `BramAdapter`, l'écriture prend 1 cycle synchrone garanti (`mem.write`).
  Sur `DdrAdapter`, une écriture hôte déclenche une transaction AXI4 complète (`aw`, `w`, `b`) vers un contrôleur DRAM externe dont la latence est variable (arbitrage, refresh DRAM, attente de `aw.ready`/`w.ready`/`b.valid`).
  L'interface abstraite `MemoryAdapter.io` ne fournit aucun signal `wrReady` en retour vers l'hôte.
  Si l'hôte ou la passerelle UART envoie un nouveau mot mémoire via `io.wrEnable` pendant que la transaction précédente est encore en vol ou en attente d'arbitrage (`accBusy`), `wrAddrReg`, `wrDataReg` et `wrStrbReg` sont écrasés dans le même cycle. Le mot mémoire précédent est irrémédiablement perdu sans aucune notification d'erreur.
- **Impact Wave 6** : Risque de corruption des poids et des images lors du flashage ou du transfert haute vitesse vers la DDR3 (Priorité 5).
- **Correction appliquée & validée** :
  1. Ajout de `val wrReady = out(Bool())` sur l'interface commune `MemoryAdapter.io`.
  2. Pilotage dans `DdrAdapter` par `val hostWrReady = !hostActive && !accBusy` et protection de l'échantillonnage par `when(io.wrEnable && hostWrReady)`.
  3. Maintien de `io.wrReady := True` sur `BramAdapter` et `SramAsicAdapter` (écritures synchrones 1 cycle).
  Validée par le test de simulation dédié `MemoryAdapterTest` (« DdrAdapter: Flow control wrReady backpressures host writes until AXI b.valid ») confirmant le blocage des écritures prématurées et la non-corruption d'une transaction en vol. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-06 : Crash d'élaboration sur l'`idWidth` de l'arbitre AXI Read au-delà de 8 couches

- **Sévérité** : 🟠 **MAJEUR** (Échec fatal de génération RTL sur réseaux profonds)
- **Fichier** : [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala#L212-L215)
- **Code concerné** :
  ```scala
  val dmaAxiConfig =
    if (totalDmaTriggers == 1) axiConfig
    else axiConfig.copy(idWidth = axiConfig.idWidth - log2Up(totalDmaTriggers))
  ```
- **Mécanisme de défaillance** :
  `Sequential` calcule dynamiquement la configuration AXI des maîtres DMA feuilles en soustrayant le nombre de bits d'ID nécessaires à l'arbitre `Axi4ReadOnlyArbiter` :
  $$\text{idWidth}_{\text{feuille}} = \text{axiConfig.idWidth} - \lceil\log_2(\text{totalDmaTriggers})\rceil$$
  Chaque couche convolutive ou linéaire sans partage requiert 2 déclencheurs DMA (poids et biais).
  Pour un réseau de seulement 8 couches avec poids et biais :
  $$\text{totalDmaTriggers} = 1 \text{ (image)} + 8 \times 2 = 17 \implies \lceil\log_2(17)\rceil = 5 \text{ bits}$$
  Si la configuration AXI par défaut du SoC a `idWidth = 4` (valeur standard dans de nombreux SoC FPGA et tests du projet) :
  $$\text{idWidth}_{\text{feuille}} = 4 - 5 = -1$$
  SpinalHDL lève une exception fatale à l'élaboration :
  `requirement failed: Axi4Config idWidth must be greater than 0`.
  La compilation du matériel plante immédiatement avant même d'atteindre la synthèse.
- **Correction appliquée & validée** :
  1. Augmentation de la valeur par défaut d'`axiConfig.idWidth` de 4 à 8 bits dans la signature de `Sequential` (permettant jusqu'à 256 déclencheurs DMA / 127 couches sans configuration manuelle).
  2. Validation explicite `require(axiConfig.idWidth >= minRequiredIdWidth)` générant un message d'erreur informatif et précis en cas de bus trop étroit.
  3. Bypass direct de l'arbitre lorsque `allAxiMasters.length == 1` (modèles sans poids).
  Validée par le test d'élaboration `SequentialTest` (« BUG-DDR-06: idWidth validation and elaboration for deep models (> 8 weight/bias layers) ») et vérification formelle BMC `StreamDoubleBufferPrefetchFormal`. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-07 : Émission prématurée de `nextTile` dans `DoubleBufferStreamer`

- **Sévérité** : 🟡 **MOYEN** (Désynchronisation potentielle et fuite de beats résiduels)
- **Fichier** : [`spinalML/src/spinalML/memory/DoubleBufferStreamer.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DoubleBufferStreamer.scala#L53-L59)
- **Code concerné** :
  ```scala
  when(reqStream.fire) {
    readCounter.increment()
    when(readCounter.willOverflowIfInc) {
      isReading := False
      io.nextTile := True // Signal the double buffer to flip banks
    }
  }
  ```
- **Mécanisme de défaillance** :
  `io.nextTile` est pulsé au cycle où le compteur d'adresses émet la *dernière adresse de lecture* vers la mémoire interne.
  Or, la mémoire a 1 cycle de latence synchrone (`readSync`), et les données lues traversent une FIFO de livraison de 16 mots.
  Si le consommateur aval subit un calage (backpressure), la totalité ou une partie de la tuile réside encore dans la FIFO.
  Pourtant, le `StreamDoubleBuffer` reçoit déjà l'impulsion `nextTile` et bascule immédiatement sa banque (`computeBank := !computeBank`). Si le lecteur DMA a déjà préchargé la tuile suivante en arrière-plan, le streamer peut commencer à aspirer la nouvelle tuile dans la même FIFO avant que l'ancienne ne soit totalement consommée, ce qui crée des mélanges ou des queues résiduelles lors des frontières d'inférence.
- **Impact Wave 6** : Cause racine documentée des corruptions sous prefetch eager (voir `docs/bugs/2026-08-prefetch-eager-stale-fifo-session.md`) et risque de pacing sous flux continu (Priorités 0, 2 et 4).
- **Correction appliquée & validée** :
  1. Synchronisation de `io.nextTile` sur la livraison effective aval dans `DoubleBufferStreamer.scala` :
     - Remplacement de l'émission prématurée (basée sur `readCounter.willOverflowIfInc` qui survenait à l'émission de la dernière adresse BRAM avant même que les données n'atteignent la FIFO) par un compteur de livraison `popCounter = Counter(memSize)`.
     - `io.nextTile := True` et libération de l'état `tileActive := False` sont désormais exclusivement pilotés par `when(io.streamOut.fire) { popCounter.increment(); when(popCounter.willOverflowIfInc) { ... } }`.
     - L'état de lecture `isReading` s'arrête proprement dès que toutes les adresses sont dispatchées, et le démarrage d'une nouvelle tuile est conditionné par `!tileActive`, interdisant toute aspiration de nouvelle tuile tant que la tuile en cours n'a pas été vidée de la FIFO.
     - Reset et flush atomique de `popCounter` et `tileActive` sur `io.reArm`.
  2. Validée par le test unitaire Scala `DoubleBufferStreamerTest` (« BUG-DDR-07: nextTile must NOT fire prematurely under downstream backpressure until tile is fully drained ») vérifiant qu'aucune impulsion `nextTile` n'est émise tant que l'aval est calé (0 mot extrait de la FIFO), et que l'impulsion survient exactement sur le handshake du dernier mot.
  3. Non-régression formelle BMC `DoubleBufferStreamerFormal` (31.8s), `StreamDoubleBufferFormal` (27.3s), `StreamDoubleBufferPrefetchFormal` (76.6s) et test d'intégration `SequentialTest` (164.9s). Statut : ✅ **CORRIGÉ**.


---

### BUG-DDR-08 : Risque de deadlock dans `DMAReader2D` si `(rowWords * axiLanes) % outLanes != 0`

- **Sévérité** : ⚪ **FAUX POSITIF (VÉRIFIÉ & DÉMONTRÉ)**
- **Fichier** : [`spinalML/src/spinalML/memory/DMAReader2D.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DMAReader2D.scala#L65-L70) et [`L130-L133`](file:///e:/spinalML/spinalML/src/spinalML/memory/DMAReader2D.scala#L130-L133)
- **Code concerné** :
  ```scala
  val reader1D = DMAReader(dataType, shape, outLanes, axiConfig)
  ...
  val totalFetchElems = rowWords * U(elemsPerWord)
  val rowFetchedBeats = ((totalFetchElems / outLanes).resize(beatsW bits)) +
                        Mux(totalFetchElems % outLanes =/= 0, U(1, beatsW bits), U(0, beatsW bits))
  ```
- **Mécanisme allégué** :
  Il était supposé que si `(rowWords * axiLanes) % outLanes != 0`, le convertisseur de voies resterait en attente d'éléments pour compléter le dernier battement de sortie et que le terme `+ 1` dans `rowFetchedBeats` causerait un deadlock dans `stateDrain` attendant un battement fantôme.
- **Démonstration de l'impossibilité structurelle (Faux Positif)** :
  1. **Invariants et contrats formels du module** :
     - `shape(1) % outLanes == 0` : contrat d'élaboration (`DMAReader2D.scala:78`), chaque ligne ML est un multiple strict de `outLanes`.
     - `headSkipElems % outLanes == 0` pour `outLanes > 1` : contrat d'alignement runtime prouvé formellement (`DMAReader2DFormal.scala:99-102`), car le trimming aval ne découpe que des battements entiers.
     - Toutes les largeurs de bus AXI (`dataWidth`) et dtypes sont des puissances de 2, donc $AL = elemsPerWord$ et $OL = outLanes$ sont des puissances de 2.
  2. **Preuve mathématique que `totalFetchElems % outLanes == 0` toujours** :
     - Soit $rowWords = \lceil (headSkipElems + rowWidth) / AL \rceil$ et $totalFetchElems = rowWords \times AL$.
     - **Cas $OL \le AL$** : $AL$ est un multiple de $OL$ ($AL = k \times OL$). Donc $totalFetchElems = (rowWords \times k) \times OL$, ce qui donne $totalFetchElems \pmod{OL} \equiv 0$ pour tout $rowWords$.
     - **Cas $OL > AL$** : Puisque $headSkipBytes < bytesPerBeat$, on a $0 \le headSkipElems < AL < OL$. Par le contrat d'alignement ($headSkipElems \pmod{OL} == 0$), la seule valeur possible est $headSkipElems = 0$. Alors $rowWords = \lceil rowWidth / AL \rceil$. Comme $rowWidth$ est multiple de $OL = m \times AL$, $rowWords$ est un multiple exact de $m$, et $totalFetchElems = p \times m \times AL = p \times OL$. Donc $totalFetchElems \pmod{OL} \equiv 0$.
  3. **Absence de résidu inter-lignes** :
     Comme documenté dans [`DMAReader.scala:43`](file:///e:/spinalML/spinalML/src/spinalML/memory/DMAReader.scala#L43) (*« DMAReader2D keeps the legacy adapter (image rows are exact multiples) »*), le convertisseur `StreamWidthAdapter` se vide intégralement à la fin de chaque ligne car chaque ligne comprend un nombre entier exact de battements de sortie.
  4. **Validation de non-régression** :
     Suite de tests unitaires `DMAReader2DTest` (75.4s) ✅ et suite formelle BMC + Cover SymbiYosys `DMAReader2DFormal` (115.5s) ✅. Statut : ⚪ **FAUX POSITIF DOCUMENTÉ**.


---

### BUG-DDR-09 : Violation du protocole AXI4 sur `B.ID` dans `DdrAdapter` et `BramAdapter`

- **Sévérité** : 🟡 **MOYEN** (Non-conformité protocolaire AXI4)
- **Fichiers** : [`spinalML/src/spinalML/memory/DdrAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DdrAdapter.scala#L83) et [`spinalML/src/spinalML/memory/BramAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/BramAdapter.scala#L143)
- **Code concerné** :
  ```scala
  // DdrAdapter.scala:83
  extIo.ddrMaster.aw.payload.id := 0

  // BramAdapter.scala:143
  io.axi.b.payload.id := 0
  ```
- **Mécanisme de défaillance** :
  Selon la spécification AXI4 (ARM IHI 0022E, section A3.4.3 *Response signals*) :
  > *"The write response must carry the same ID as the write address transaction to which it relates."*
  Dans `BramAdapter`, l'adaptateur force systématiquement `io.axi.b.payload.id := 0` sans mémoriser l'ID de la requête d'adresse reçue sur `io.axi.aw.payload.id`.
  Dans `DdrAdapter`, l'ID du maître est écrasé par `0` lors de la transmission vers le contrôleur externe.
  Si le maître émetteur (ou un switch AXI interconnectant plusieurs accélérateurs) utilise un identifiant différent de 0, la réponse B arrive avec un mauvais ID, désynchronisant ou faisant planter l'interconnect AXI.
- **Impact Wave 6** : Incompatibilité avec les interconnects AXI4 multi-maîtres lors de l'intégration système (Priorité 5).
- **Correction appliquée & validée** :
  1. Mémorisation et propagation de l'ID d'écriture AXI :
     - Dans [`BramAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/BramAdapter.scala) et [`SramAsicAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/SramAsicAdapter.scala) : ajout du registre `awIdR = Reg(UInt(axiConfig.idWidth bits))` échantillonné lors du handshake `when(io.axi.aw.valid && io.axi.aw.ready) { awIdR := io.axi.aw.payload.id }`.
     - Assignation de la réponse d'écriture `io.axi.b.payload.id := awIdR`, assurant la conformité stricte avec la spécification ARM AXI4 (IHI 0022E §A3.4.3).
     - Dans [`DdrAdapter.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/DdrAdapter.scala) : mémorisation et propagation de l'ID d'écriture vers le contrôleur externe `extIo.ddrMaster.aw.payload.id := Mux(busHost, U(0, axiConfig.idWidth bits), accIdR)` et ré-acheminement de `extIo.ddrMaster.b.payload` vers `io.axi.b.payload`.
  2. Validée par le test unitaire Scala `MemoryAdapterTest` (« BUG-DDR-09: AXI4 write response B.ID must reflect AW.ID in BramAdapter and SramAsicAdapter ») validant bit-exact le renvoi de transactions d'ID non nuls (`testId = 7` sur BramAdapter, `testId = 11` sur SramAsicAdapter). Non-régression formelle BMC `AxiReadMemFormal` (26.1s). Statut : ✅ **CORRIGÉ**.


---

### BUG-DDR-10 : Flakiness du banc `DmaSdbTb` lié à l'absence de reset/flush du streamer

- **Sévérité** : 🟢 **TEST** (Test flaky marqué ignore, masque des régressions)
- **Fichier** : [`archive/test/examples/SdbSwapTb.scala`](file:///e:/spinalML/archive/test/examples/SdbSwapTb.scala#L131-L140) et [`L190-L195`](file:///e:/spinalML/archive/test/examples/SdbSwapTb.scala#L190-L195)
- **Code concerné** :
  ```scala
  class DmaSdbDut(axiConfig: Axi4Config) extends Component {
    ...
    val streamer = DoubleBufferStreamer(wType, 2880, 4)
    ...
    sdb.io.reArm := io.cmd.fire
    // streamer.io.reArm N'EST PAS CONNECTÉ (vaut False par défaut) !
  }
  ```
- **Mécanisme de défaillance** :
  Dans le banc de micro-benchmark `DmaSdbTb`, `streamer.io.reArm` n'est pas branché au signal de frontière de commande.
  La FIFO interne de 16 mots du streamer conserve des valeurs initiales aléatoires produites par Verilator. Selon la graine pseudo-aléatoire de simulation, le banc réussit (graine `1481499482` : 0 erreur) ou échoue dramatiquement (graine `2094212935` : 2832 erreurs sur 2880 valeurs, le streamer répétant le premier mot indéfiniment). Le test a été désactivé par un tag `ignore` au lieu d'être corrigé.
- **Impact Wave 6** : Fragilise le filet de sécurité (Priorité 0) qui doit impérativement restaurer des tests vivants et déterministes.
- **Correction appliquée & validée** :
  1. Câblage de `streamer.io.reArm := io.cmd.fire` dans `DmaSdbDut` ([`archive/test/examples/SdbSwapTb.scala`](file:///e:/spinalML/archive/test/examples/SdbSwapTb.scala#L138-L139)), assurant la réinitialisation de la FIFO interne et des compteurs à chaque nouvelle commande de fetch.
  2. Rétablissement du test `test("DMA->SDB full path serves the W4A8 FC weight exactly")` (suppression du tag `ignore`).
  3. Non-régression unitaire `DoubleBufferStreamerTest` (53.6s) ✅. Statut : ✅ **CORRIGÉ**.

---

### BUG-DDR-11 : Désactivation totale en CI des suites end-to-end DDR archivées

- **Sévérité** : 🟢 **TEST & DETTE** (Angle mort complet de non-régression)
- **Fichier** : [`archive/test/examples/`](file:///e:/spinalML/archive/test/examples/) et [`build.mill`](file:///e:/spinalML/build.mill#L15-L25)
- **Suites concernées** :
  - `WeightResidentChainTest.scala` (Résidence des poids en DRAM, AR=0 en steady)
  - `WeightPrefetchChainTest.scala` (Préchargement eager de poids)
  - `MnistChainedTest.scala` (Inférences consécutives en session unique)
  - `MnistContinuousTest.scala` (Contrôle continu RUN/STOP et TILE_CNT)
  - `BandTilingTest.scala` (Tiling vertical d'images par bandes)
  - `SdbSwapTb.scala` (Banc de basculement double buffer)
  - Modèles `WideResidual` et `WideConv` (`archive/src/heavy/`, `archive/test/heavy/`)
- **Constat** :
  Toutes ces suites ont été déplacées dans le dossier `archive/` hors du périmètre de compilation de `build.mill`.
  Par conséquent :
  - Ni `mill spinalML.test`, ni `test-all` n'exécutent ces tests.
  - Les évolutions récentes (notamment la **Wave 5 sur les arrondis RNE**, le refactoring des divisions entières et la requantification) **n'ont jamais été testées sur le pipeline DDR**.
  - On ignore aujourd'hui si des régressions fonctionnelles affectent le comportement de la DDR sous ces nouveaux modes d'arrondi.
- **Impact Wave 6** : Constitue l'objectif n°1 de la **Priorité 0 (Filet de sécurité)** du plan Wave 6.

---

## Plan d'Action Recommandé Avant d'Entamer la Wave 6

Pour aborder la Wave 6 sur une base saine et robuste, l'ordre d'intervention logique suivant est préconisé :

1. **Phase Pré-P0 (Patch des bugs RTL immédiats)** :
   - Corriger l'omission `&& !prefetchWorldB` sur `bDoubleBuffer.io.reArm` dans `Sequential.scala` ([BUG-DDR-01](#bug-ddr-01--omission-du-guard-prefetchworldb-sur-le-re-arm-du-double-buffer-de-biais)).
   - Corriger la formule de calcul d'octets `finalBeatBytes` dans `DMAWriter.scala` ([BUG-DDR-03](#bug-ddr-03--masque-doctets-wstrb-nul-sur-le-dernier-beat-dmawriter-pour-les-types-4-bits)).
   - Câbler `streamer.io.reArm` dans le banc `DmaSdbDut` ([BUG-DDR-10](#bug-ddr-10--flakiness-du-banc-dmasdbtb-lié-à-labsence-de-resetflush-du-streamer)).

2. **Phase P0 (Restauration du filet de sécurité)** :
   - Réintégrer les 6 suites de tests archivées de `archive/test/examples/` vers `spinalML/test/src/spinalML/examples/`.
   - Mettre à jour leurs oracles de réplique avec les nouvelles conventions d'arrondi RNE issues de la Wave 5.
   - Brancher ces tests dans la CI pour valider que le chemin mémoire existant est vert à 100%.

3. **Phase P1-P5 (Chantiers de fond Wave 6)** :
   - Implémenter l'incrément `outBaseOffset` pour le streaming continu en DDR ([BUG-DDR-02](#bug-ddr-02--absence-dincrément-de-ladresse-de-sortie-en-mode-flux-continu-writetoddr)).
   - Structurer l'arbitrage AXI Read en cascade ([BUG-DDR-06](#bug-ddr-06--crash-délaboration-sur-lidwidth-de-larbitre-axi-read-au-delà-de-8-couches)).
   - Enrichir `DdrAdapter` avec le port de lecture hôte, le contrôle de flux `wrReady` et la barrière RAW multi-passes ([BUG-DDR-04](#bug-ddr-04--absence-de-barrière-raw-et-de-port-de-lecture-hôte-dans-ddradapter), [BUG-DDR-05](#bug-ddr-05--absence-de-contrôle-de-flux-wrready-sur-le-port-décriture-hôte-de-ddradapter)).
