# Plan de Refactorisation Dual-Target (FPGA & ASIC) — Architecture à 3 Couches

> **Date** : Septembre 2026  
> **Auteur** : Léonard Adamo (Juste-Leo2) & Antigravity  
> **Branche active** : `refactoring`  
> **Objectif** : Assainir et découpler le socle SpinalML existant pour garantir une portabilité stricte entre cibles FPGA (Gowin GW2A-18 / Tang Primer 20K, Xilinx) et cibles ASIC (OpenLane, SkyWater 130nm, GF180MCU) sans duplication de code.

---

## 1. L'Architecture Cible : Le « Sandwich à 3 Couches »

Pour qu'une unique base de code serve à la fois d'émulateur matériel bit-exact sur FPGA et de RTL prêt pour le placement-routage silicium (ASIC), les responsabilités sont strictement partitionnées en trois couches étanches :

```
┌────────────────────────────────────────────────────────────────────────┐
│  COUCHE 1 : CŒUR ALGORITHMIQUE SPINALML (100% Portable RTL)           │
│  - Flux de données valid/ready (`Stream[Tensor]`), contre-pression     │
│  - Opérateurs arithmétiques génériques (Conv2D, Matmul, Activations)   │
│  - Pipeline SIMD universel (`lanes: Int` paramétrable)                │
│  - Trait polymorphe `Target` (Target.FPGA vs Target.ASIC)              │
│  - ZÉRO primitive propriétaire constructeur                            │
│  - ZÉRO adresse mémoire physique ou taille de BRAM codée en dur        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Interface neutre (MemoryBus / AXI4)
┌───────────────────────────────────▼────────────────────────────────────┐
│  COUCHE 2 : ADAPTATEUR MÉMOIRE ABSTRAIT (`MemoryAdapter`)              │
│  - Interface unique pour commandes/réponses de lecture et d'écriture  │
│  - Implémentation BramAdapter     : BRAM interne FPGA & co-sim C++     │
│  - Implémentation DdrAdapter      : Contrôleur DDR3 physique (Tang)   │
│  - Implémentation SramAsicAdapter : Macros SRAM OpenRAM (SKY130/GF180) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Broches physiques & horloges
┌───────────────────────────────────▼────────────────────────────────────┐
│  COUCHE 3 : PHYSIQUE & INTÉGRATION BOÎTIER                             │
│  - FPGA : Contraintes de broches (.cst / .xdc), reset bitstream BOOT   │
│  - ASIC : Padring, Leadframe, broche reset externe `rst_n`, CTS       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Audit Critique de la Phase 1 de la Roadmap : Verdict Réel

L'audit détaillé du code source confirme la validité de la roadmap à **90%**, tout en rectifiant plusieurs détails d'implémentation et en révélant des briques existantes à réutiliser.

### Tableau Récapitulatif

| Refactoring | Diagnostic Code Réel | Statut Roadmap | Effort Estimé | Fichiers Clés |
| :--- | :--- | :--- | :--- | :--- |
| **1.1 Line Buffers `im2col`** | Shift register géant `Vec(Reg)` (82k FFs) | **VRAI** (erreur de chemin corrigée, `LineBuffer2D` déjà codé ailleurs) | 1 à 2 jours | `ops/im2col.scala`, `poolings/maxpool2d.scala` |
| **1.2 Multiplieur / `Target`** | `DspMul` a déjà un fallback mais pas de trait `Target` explicite | **VRAI** (imprécision sur `Conv2D` corrigée) | 0.5 à 1 jour | `dsp/DspMul.scala`, `dsp/Target.scala` |
| **1.3 `MemoryAdapter`** | `AxiReadMem` (32 Ko, adresses fixes) soudé dans `UartSoC` | **100% VRAI** | 1 à 2 jours | `io/AxiReadMem.scala`, `io/UartSoC.scala`, `io/MemoryAdapter.scala` |
| **1.4 Paramétrisation `lanes`** | Goulot d'étranglement `lanes = 1` forcé partout dans `Sequential` | **100% VRAI** | 2 à 3 jours | `nn/Sequential.scala`, `layers/Conv2D.scala`, `ops/im2col.scala` |
| **1.5 AXI Master Write (`DMAWriter`)** | Canaux AXI `aw`/`w` forcés à `False` dans `Accelerator` | **100% VRAI** | 1 à 2 jours | `nn/Accelerator.scala`, `memory/DMAWriter.scala` |

---

## 3. Détail Opérationnel des 5 Refactorings

### Refactoring 1.1 : Migration des Line Buffers de `im2col.scala` vers `Mem.readSync`

* **Fichier réel** : [`spinalML/src/spinalML/ops/im2col.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/im2col.scala) (et non `layers/im2col.scala`).
* **Constat dans le code** :
  ```scala
  // spinalML/src/spinalML/ops/im2col.scala:L55-59
  val lineBuffers = for (i <- 0 until K - 1) yield new Area {
    val regs = Vec(Reg(dataType), W * C)
    val pop = regs(W * C - 1)
  }
  ```
  Les registres sont décalés un par un à chaque cycle (L116-120). Pour une couche de vision $160 \times 160 \times 32$, cela synthétise **81 920 bascules D (FFs)**. Le FPGA Gowin GW2A-18 sature à 15 552 FFs ; en ASIC, c'est une débauche de surface intenable.
* **Solution à adopter** :
  * Extraire le composant [`LineBuffer2D[T]`](file:///e:/spinalML/spinalML/src/spinalML/poolings/maxpool2d.scala#L16-L35) actuellement dans `maxpool2d.scala` pour le placer dans `spinalML.memory.LineBuffer2D`.
  * Ce composant utilise déjà un `Mem(dataType, depth)` synchrone circulaire avec pointeur `Counter(depth)` et `readSync`.
  * Remplacer `lineBuffers` dans `im2col.scala` par des instances de `LineBuffer2D`.
  * **Point d'attention** : `readSync` ayant 1 cycle de latence synchrone, décaler d'un cycle la validation d'assemblage des colonnes dans la FSM d'`im2col`.
* **Critère de succès** :
  * Réduction drastique des FFs en synthèse Yosys (inférence de BRAM words).
  * Exécution bit-exacte des tests : `.venv/Scripts/python.exe cli/main.py mill spinalML.test.testOnly spinalML.ops.Im2ColTest`.

---

### Refactoring 1.2 : Abstraction Multiplieur et Trait `Target`

* **Fichiers impactés** : [`spinalML/src/spinalML/dsp/DspMul.scala`](file:///e:/spinalML/spinalML/src/spinalML/dsp/DspMul.scala), [`spinalML/src/spinalML/dsp/DspConfig.scala`](file:///e:/spinalML/spinalML/src/spinalML/dsp/DspConfig.scala), nouveau `spinalML/src/spinalML/dsp/Target.scala`.
* **Constat dans le code** :
  * `DspMul.scala` sait déjà faire de la multiplication comportementale pure `(a * b)` lorsqu'il n'est pas sur Gowin physique.
  * En revanche, le choix de la cible est implicite, basé sur des variables d'environnement (`SPINALML_TARGET`), sans typage strict.
  * Il n'y a pas d'entité `Target.ASIC` permettant de configurer explicitement les étages de retiming/pipeline standard pour OpenROAD.
* **Solution à adopter** :
  * Définir un trait polymorphe scellé :
    ```scala
    sealed trait Target
    object Target {
      case class FPGA(family: FpgaFamily = FpgaFamily.Gowin, useHardDsp: Boolean = true) extends Target
      case class ASIC(pdk: PdkFamily = PdkFamily.Sky130, pipelineStages: Int = 1) extends Target
      case object Simulation extends Target
    }
    ```
  * Injecter `target: Target` dans `SpinalMLConfig` et le propager proprement vers `DspMul` (qui devient `HardwareMul`).
  * En mode `Target.ASIC`, interdire formellement toute boîte noire constructeur et générer des multiplieurs comportementaux synchronisés avec retiming.
* **Critère de succès** :
  * Compilation bit-exacte garantie sur Verilator.
  * Génération de Verilog propre sans macro Gowin en configuration `Target.ASIC`.

---

### Refactoring 1.3 : Extraction de l'Adaptateur Mémoire Abstrait (`MemoryAdapter`)

* **Fichiers impactés** : [`spinalML/src/spinalML/io/AxiReadMem.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/AxiReadMem.scala), [`spinalML/src/spinalML/io/UartSoC.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/UartSoC.scala), [`spinalML/src/spinalML/nn/Accelerator.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala).
* **Constat dans le code** :
  * `AxiReadMem` possède des adresses codées en dur (`imgBase = 0x10000`, `weightBase = 0x20000`) et alloue une BRAM fixe de 4096 mots de 64 bits (32 Ko).
  * `UartSoC` instancie directement `AxiReadMem`, rendant impossible le basculement vers la DDR3 de la Tang Primer 20K ou des macros SRAM OpenRAM pour un ASIC.
* **Solution à adopter** :
  * Créer le trait abstrait Couche 2 :
    ```scala
    trait MemoryAdapter extends Component {
      val io: Bundle {
        val axiSlave: Axi4ReadOnly // ou interface native
        // ports physiques vers la techno sous-jacente
      }
    }
    ```
  * Créer `BramAdapter` (BRAM interne paramétrable en profondeur et en plages d'adresses).
  * Préparer les stubs pour `DdrAdapter` (DDR3 Tang Primer) et `SramAsicAdapter` (macros OpenRAM).
  * Dans `UartSoC` et `Accelerator`, découpler l'accès mémoire pour accepter n'importe quel `MemoryAdapter`.
* **Critère de succès** :
  * Le SoC fonctionne indifféremment sur BRAM ou DDR sans toucher à une ligne du cœur d'accélération.

---

### Refactoring 1.4 : Propagation Paramétrique Universelle de `lanes`

* **Fichiers impactés** : [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala), [`spinalML/src/spinalML/layers/Conv2D.scala`](file:///e:/spinalML/spinalML/src/spinalML/layers/Conv2D.scala), [`spinalML/src/spinalML/ops/im2col.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/im2col.scala), [`spinalML/src/spinalML/io/UartSoC.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/UartSoC.scala).
* **Constat dans le code** :
  * Dans `Sequential.scala` (L528-532, 558-560), la sortie de presque chaque couche est ré-emballée en `lanes = 1` avec `repack(..., 1)`.
  * `Conv2D.scala` force `io.x` et `io.y` avec `lanes = 1`.
  * `im2col.scala` force l'entrée avec `lanes = 1`.
  * `UartSoC.scala` extrait `payload(0)` et tronque à 8 bits.
* **Solution à adopter** :
  * Rendre le paramètre `lanes: Int` configurable pour chaque couche.
  * Adapter `im2col.scala` pour accepter une entrée multi-canaux / multi-pixels en parallèle, ou insérer automatiquement un `RepackOp` d'adaptation de largeur uniquement quand nécessaire.
  * Adapter la sérialisation de sortie dans `UartSoC` pour décharger les `lanes` séquentiellement vers l'UART.
* **Critère de succès** :
  * Validation bit-exacte pour `lanes = 1, 2, 4` sur les suites de tests universels.

---

### Refactoring 1.5 : Canaux d'Écriture AXI Master (`DMAWriter`)

* **Fichiers impactés** : [`spinalML/src/spinalML/nn/Accelerator.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala), nouveau `spinalML/src/spinalML/memory/DMAWriter.scala`.
* **Constat dans le code** :
  * Dans `Accelerator.scala` (L68-72), les canaux d'écriture AXI sont neutralisés (`aw.valid := False`, `w.valid := False`).
  * L'accélérateur ne sait que lire depuis la mémoire externe et cracher les résultats sur un flux sortant.
  * Pour un système ASIC ou SoC autonome, écrire les activations intermédiaires (spilling) ou stocker la carte de sortie en mémoire vive est indispensable.
* **Solution à adopter** :
  * Développer le pendant de `DMAReader` : `DMAWriter[T]`, capable de regrouper les streams `Tensor` en bursts AXI4 `INCR` (jusqu'à 256 beats) avec gestion du découpage aux frontières de 4 Ko.
  * Connecter les canaux `aw`, `w` et `b` dans `Accelerator.scala`.
* **Critère de succès** :
  * Banc de test de co-simulation Verilator écrivant un tenseur complet en mémoire et relisant les données pour vérification d'intégrité bit-exacte.

---

## 4. Dettes "FPGA-like" Additionnelles Découvertes

Au-delà de la roadmap initiale, trois détails propres au monde FPGA doivent être isolés pour réussir un tapeout ASIC :

1. **Le Reset `BOOT` dans `UartSoC.scala:L40`** :
   * Le code utilise `ClockDomainConfig(resetKind = BOOT)` avec un compteur de délai.
   * Sur FPGA, c'est l'automate de chargement du bitstream qui gère ce démarrage.
   * **Sur ASIC, le reset `BOOT` n'existe pas.** Il faut une broche externe `rst_n` (actif bas, asynchrone) et un réseau d'arbre de reset (CTS) dédié.
2. **L'initialisation implicite des mémoires (`mem.init`)** :
   * Les BRAM de FPGA sont initialisées à zéro lors du chargement du bitstream.
   * Les macros SRAM d'un ASIC (générées par OpenRAM) démarrent dans un état **aléatoire et indéterminé**.
   * Toute logique qui présuppose que la mémoire est initialisée à zéro au démarrage doit être revue pour que l'initialisation soit explicite ou que les premières lectures attendent la première passe d'écriture.
3. **Le Déploiement 100% Spatial** :
   * `Sequential.scala` instancie physiquement chaque couche l'une après l'autre en silicium.
   * Pour un petit réseau de 3 couches (MNIST), cela passe. Pour un modèle de vision type YOLO (60 couches), la surface en ASIC explose.
   * C'est la justification de la Phase 3 : basculer vers un **Cœur Replié (Folded Core)** réutilisant le même bloc physique pour exécuter séquentiellement les différentes couches du réseau.

---

## 5. Question Technique : Le Nombre Magique dans `examples/Mnist/Model.scala`

### Analyse du problème
Dans [`examples/Mnist/Model.scala:L46`](file:///e:/spinalML/examples/Mnist/Model.scala#L46), on trouve la ligne suivante :
```scala
// 2. Transition from Integer domain to Floating-Point domain (FP8 E4M3)
Cast(FP8_E4M3(), scales = Seq(0.08544921875)),
```

* **D'où vient ce nombre ?**  
  $0.08544921875 = \frac{175}{2048} \approx \frac{175}{2^{11}}$.  
  C'est le facteur d'échelle de déquantification (dequantization scale) calculé lors de l'entraînement PyTorch en quantification mixte w4a8.  
  Le début du modèle (Conv2D) tourne en arithmétique entière INT4 / INT16, tandis que la fin du modèle (Linear) tourne en flottant FP8. Le composant `CastOp` doit donc multiplier la valeur entière par ce facteur pour la faire atterrir sur la grille dynamique du FP8 :  
  $$x_{\text{FP8}} = \text{FloatML}(x_{\text{INT16}}) \times 0.08544921875$$

* **Est-ce que notre plan de refactorisation règle ce problème ?**  
  **OUI, et voici comment :**
  1. **Disparition dans les architectures homogènes** :  
     Les modèles de vision cibles pour l'ASIC (comme YOLOv8n) s'exécutent en **quantification homogène** (tout en INT8 avec `Requantize`, ou tout en FP8/BF16 de bout en bout). Il n'y a donc plus de saut de domaine INT $\to$ Float au milieu du pipeline, et ces coefficients de cast disparaissent purement et simplement.
  2. **Découplage de la topologie matérielle (Phase 3 & Folded Core)** :  
     Même dans le cas où un modèle requiert des facteurs de mise à l'échelle (par exemple pour la requantification INT32 $\to$ INT8 ou le scaling de BatchNorm), **ceux-ci ne doivent JAMAIS être codés en dur dans la topologie Scala du réseau**.  
     Ils doivent faire partie des **poids / métadonnées** stockés en mémoire (DDR / SRAM) ou configurés par le CPU hôte via les registres CSR (`AxiLite4SlaveFactory`).  
     Le cœur matériel exécute alors l'opération avec un registre de gain programmable, éliminant tout nombre magique dans le code source Scala.

---

## 6. Plan d'Action Ordonné & Estimation Temporelle

Le refactoring complet de la Phase 1 représente environ **800 à 1 200 lignes de Scala**, réalisable en **1 à 2 semaines de travail**.

```mermaid
graph TD
    A[Étape 1: LineBuffer2D + im2col] -->|FFs réduits, synthèse BRAM| B[Étape 2: Target Trait & HardwareMul]
    B -->|Découplage vendor IP| C[Étape 3: MemoryAdapter & Isolation SoC]
    C -->|Mémoire swappable BRAM/DDR/ASIC| D[Étape 4: Propagation paramétrique lanes]
    D -->|Haut débit SIMD| E[Étape 5: DMAWriter AXI Master Write]
```

### Ordre d'exécution recommandé :

1. **Jalon 1 (Jours 1-2) : Débloquer la mémoire des convolutions**
   - Extraire `LineBuffer2D` dans `spinalML.memory`.
   - Migrer `im2col.scala` vers `LineBuffer2D` avec `Mem.readSync`.
   - Valider la non-régression avec `python cli/main.py mill spinalML.test.testOnly spinalML.ops.Im2ColTest`.

2. **Jalon 2 (Jours 3-4) : Abstraire la cible matérielle**
   - Créer `spinalML.dsp.Target` (`Target.FPGA`, `Target.ASIC`, `Target.Simulation`).
   - Mettre à jour `SpinalMLConfig` et `DspMul.scala` pour respecter ce contrat.

3. **Jalon 3 (Jours 5-7) : Découpler la mémoire et le SoC**
   - Créer le trait `MemoryAdapter`.
   - Isoler `AxiReadMem` en tant que `BramAdapter`.
   - Nettoyer `UartSoC` pour qu'il reçoive un `MemoryAdapter` polymorphe.

4. **Jalon 4 (Jours 8-10) : Paramétrer `lanes` et implémenter `DMAWriter`**
   - Généraliser `lanes: Int` dans les bundles d'E/S de `Conv2D` et `Sequential`.
   - Écrire `DMAWriter.scala` et connecter les canaux d'écriture dans `Accelerator.scala`.
   - Valider sur la suite de tests complète.
