# Audit Technique Complet : Lacunes et Verrous pour l'Inférence de Modèles de Vision Modernes (YOLO) dans SpinalML

Ce document dresse un état des lieux exhaustif et sans complaisance de l'ensemble du projet **SpinalML** (côté matériel Scala/SpinalHDL, architecture SoC/mémoire, et outillage Python/CLI). Il identifie précisément tout ce qui manque ou bloque pour exécuter des modèles de vision modernes (familles **YOLOv5 / YOLOv8 / YOLOv11**, ResNet, MobileNet), détaille les problèmes identifiés dans le code actuel, confronte les propositions de réutilisation matérielle à leurs véritables coûts d'ingénierie, évalue factuellement le potentiel de la structure existante vers une synthèse silicium dédiée (ASIC), et définit la feuille de route stratégique « Dual-Target » (FPGA & ASIC).

---

## Sommaire

1. [Vue d'ensemble : L'écart entre MNIST et YOLO](#1-vue-densemble--lécart-entre-mnist-et-yolo)
2. [Lacunes Matérielles et Primitives Scala (RTL)](#2-lacunes-matérielles-et-primitives-scala-rtl)
   - [2.1 Convolutions 2D : Absence de Padding (`same`)](#21-convolutions-2d--absence-de-padding-same)
   - [2.2 Convolutions 2D : Absence de Stride ($stride > 1$)](#22-convolutions-2d--absence-de-stride-stride--1)
   - [2.3 Explosion des Ressources : Line Buffers en Bascules (Flip-Flops) dans `im2col`](#23-explosion-des-ressources--line-buffers-en-bascules-flip-flops-dans-im2col)
   - [2.4 Couche d'Upsampling / Interpolation Inexistante](#24-couche-dupsampling--interpolation-inexistante)
   - [2.5 Concaténation de Tensors : Interdiction du Canal (`axis != 0`)](#25-concaténation-de-tensors--interdiction-du-canal-axis--0)
   - [2.6 Séparation de Canaux (`Split` / `Slice`) Non Exposée](#26-séparation-de-canaux-split--slice-non-exposée)
   - [2.7 Fonctions d'Activation Modernes : Absence de SiLU / Swish](#27-fonctions-dactivation-modernes--absence-de-silu--swish)
   - [2.8 MaxPool2D : Pas de Padding pour le SPPF](#28-maxpool2d--pas-de-padding-pour-le-sppf)
   - [2.9 Convolutions Séparables et Groupées (Depthwise Conv)](#29-convolutions-séparables-et-groupées-depthwise-conv)
3. [Verrous d'Architecture Système et de Mémoire](#3-verrous-darchitecture-système-et-de-mémoire)
   - [3.1 Le Mur de Surface : Limite du Pipeline Déroulé Statique](#31-le-mur-de-surface--limite-du-pipeline-déroulé-statique)
   - [3.2 Le Mur de BRAM : `TapBuffer` et les Connexions Résiduelles Géantes](#32-le-mur-de-bram--tapbuffer-et-les-connexions-résiduelles-géantes)
   - [3.3 Le Mur Mémoire : AXI en Lecture Seule et Absence de Contrôleur DDR3/PSRAM](#33-le-mur-mémoire--axi-en-lecture-seule-et-absence-de-contrôleur-ddr3psram)
   - [3.4 Topologie Multi-Têtes : Absence de Sorties Multiples](#34-topologie-multi-têtes--absence-de-sorties-multiples)
4. [Lacunes Côté Python, CLI et Écosystème](#4-lacunes-côté-python-cli-et-écosystème)
   - [4.1 Absence d'Importateur ONNX / PyTorch Automatisé](#41-absence-dimportateur-onnx--pytorch-automatisé)
   - [4.2 Fragilité du CLI : Parsing par Regex du Code Source Scala](#42-fragilité-du-cli--parsing-par-regex-du-code-source-scala)
   - [4.3 Absence de Framework de Quantification Standardisé (PTQ/QAT)](#43-absence-de-framework-de-quantification-standardisé-ptqqat)
   - [4.4 Débit de la Liaison Hôte (UART à 115200 bauds)](#44-débit-de-la-liaison-hôte-uart-à-115200-bauds)
   - [4.5 Découpage Détection : Post-Processing et NMS](#45-découpage-détection--post-processing-et-nms)
   - [4.6 Couverture des Tests Unitaires et de Non-Régression](#46-couverture-des-tests-unitaires-et-de-non-régression)
5. [Tableau Synthétique des Impacts et Priorités](#5-tableau-synthétique-des-impacts-et-priorités)
6. [Démystification des Solutions de Réutilisation & Réalité Physique](#6-démystification-des-solutions-de-réutilisation--réalité-physique)
   - [6.1 Déconstruction Critique des 4 Approches de Réutilisation](#61-déconstruction-critique-des-4-approches-de-réutilisation)
   - [6.2 La Vérité Architecturale : Ce Sont Tous des NPU Déguisés](#62-la-vérité-architecturale--ce-sont-tous-des-npu-déguisés)
   - [6.3 Le Plafond Physique Infranchissable du Tang Primer 20K](#63-le-plafond-physique-infranchissable-du-tang-primer-20k)
   - [6.4 Diagnostic Stratégique et Arbitrage](#64-diagnostic-stratégique-et-arbitrage)
7. [L'Approche du Compilateur Silicium Dédié (Model-Tailored Architecture)](#7-lapproche-du-compilateur-silicium-dédié-model-tailored-architecture)
   - [7.1 En quoi est-ce une approche conceptuellement différente ?](#71-en-quoi-est-ce-une-approche-conceptuellement-différente)
   - [7.2 Est-ce que cette approche règle les problèmes du Point 6 ?](#72-est-ce-que-cette-approche-règle-les-problèmes-du-point-6)
   - [7.3 La structure actuelle de SpinalML a-t-elle le potentiel réel ?](#73-la-structure-actuelle-de-spinalml-a-t-elle-le-potentiel-réel)
   - [7.4 Synthèse et Faisabilité Factuelle](#74-synthèse-et-faisabilité-factuelle)
8. [La Stratégie Dual-Target (FPGA & ASIC) : Du Sandwich Théorique au Plan Réel](#8-la-stratégie-dual-target-fpga--asic--du-sandwich-théorique-au-plan-réel)
   - [8.1 Où se situe la frontière exacte entre FPGA et ASIC ?](#81-où-se-situe-la-frontière-exacte-entre-fpga-et-asic)
   - [8.2 L'Architecture Sandwich à 3 Couches](#82-larchitecture-sandwich-à-3-couches)
   - [8.3 L'Audit Réel : Pourquoi le code actuel est à ~20% et non à 80%](#83-laudit-réel--pourquoi-le-code-actuel-est-à-20-et-non-à-80)
   - [8.4 La Méthode « Faire les Deux » en 4 Étapes Séquencées](#84-la-méthode-faire-les-deux-en-4-étapes-séquencées)

---

## 1. Vue d'ensemble : L'écart entre MNIST et YOLO

Aujourd'hui, SpinalML valide avec succès un CNN jouet de type **MNIST** :
- Image d'entrée : $28 \times 28 \times 1$ (niveaux de gris, ~784 octets).
- Poids : ~7 Ko (1 Conv2D $5 \times 5$, 1 Linear).
- Topologie : purement séquentielle en ligne droite sans branchements ni skips complexes.
- Mémoire : logée intégralement dans la BRAM interne du FPGA (~32 Ko via `AxiReadMem`).

À l'inverse, un modèle de vision moderne tel que **YOLOv8n** (la plus petite variante nano) requiert :
- Image d'entrée : $640 \times 640 \times 3$ RGB (ou au minimum $320 \times 320 \times 3$), soit **300 Ko à 1.2 Mo** par image.
- Poids : **3.2 Millions de paramètres** (~3.2 Mo en INT8, 1.6 Mo en INT4).
- Nombre de couches : ~168 couches et opérations élémentaires (22 convolutions, blocs C2f à connexions résiduelles multiples, SPPF, FPN/PANet avec upsampling $2\times$ et concaténations multi-échelles).
- Mémoire intermédiaire : les cartes d'activation internes des premiers étages dépassent **800 Ko à 2 Mo**, ce qui est supérieur à la mémoire interne totale de la quasi-totalité des FPGA d'entrée/milieu de gamme (< 100 Ko de BRAM sur un Tang Primer 20K).
- Sorties : 3 têtes de détection simultanées ($80 \times 80$, $40 \times 40$, $20 \times 20$) pour détecter les objets petits, moyens et grands.

Pour passer de MNIST à YOLO, plusieurs verrous doivent être levés.

---

## 2. Lacunes Matérielles et Primitives Scala (RTL)

### 2.1 Convolutions 2D : Absence de Padding (`same`)

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/layers/Conv2D.scala`](file:///e:/spinalML/spinalML/src/spinalML/layers/Conv2D.scala#L15-L16)
  - [`spinalML/src/spinalML/nn/LayerSpec.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/LayerSpec.scala#L47-L48)
  - [`spinalML/src/spinalML/ops/im2col.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/im2col.scala#L41-L42)
- **Problématique** :
  Le calcul de dimension est codé en dur :
  ```scala
  val H_out = H - K + 1
  val W_out = W_in - K + 1
  ```
  Il n'existe aucun paramètre de padding. Toutes les convolutions actuelles sont en mode `valid`.
  Dans YOLO et ResNet, quasiment **100% des convolutions $3 \times 3$ utilisent `padding = 1`** afin que la dimension spatiale de sortie soit strictement identique à la dimension d'entrée ($H_{out} = H, W_{out} = W$). Sans cela, la taille de l'image diminue de 2 pixels à chaque couche, empêchant l'alignement dimensionnel indispensable aux connexions résiduelles (skips) et aux concaténations.
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Ajouter un paramètre `pad: Int = 0` dans `Conv2D` et `im2col`. Dans `im2col.scala`, le générateur de fenêtres glissantes doit simuler un bord virtuel : quand le compteur de pixel $x$ ou $y$ se situe dans la bordure de padding ($x < pad$ ou $x \ge W + pad$), la valeur émise dans la colonne de convolution doit être forcée à zéro (`B(0)` ou valeur nulle du type) au lieu de lire le flux d'entrée.
  - *Points d'impact* : `Im2ColOp.scala`, `Conv2DLayer.scala`, `LayerSpec.scala`, `test_conv2d.py`, répliques logicielles.

---

### 2.2 Convolutions 2D : Absence de Stride ($stride > 1$)

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/layers/Conv2D.scala`](file:///e:/spinalML/spinalML/src/spinalML/layers/Conv2D.scala)
  - [`spinalML/src/spinalML/ops/im2col.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/im2col.scala#L71-L72)
- **Problématique** :
  `Conv2D` et `im2col` n'ont aucun paramètre `stride`. Le pas de déplacement spatial est figé à 1.
  Or, dans tous les modèles modernes (YOLOv5/v8, ResNet), le sous-échantillonnage spatial (diviser la résolution par 2) est réalisé par des **convolutions avec $stride = 2$** (strided convolutions), et non par des couches de pooling. Sans $stride = 2$, il est impossible de descendre dans la hiérarchie pyramidale des échelles ($P1 \to P2 \to P3 \to P4 \to P5$).
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Dans `Im2ColOp`, ajouter un paramètre `stride: Int = 1`. Lorsque $stride = 2$, la machine d'état ne doit émettre une fenêtre valide `io.c.stream.valid := True` que lorsque `(x % stride === 0) && (y % stride === 0)`. Pour les pixels intermédiaires, le flux d'entrée est ingéré dans les line buffers mais la génération de fenêtre de sortie est ignorée.
  - *Points d'impact* : `Im2ColOp.scala`, gestion de la contre-pression (`ready`/`valid`), synchronisation temporelle avec le `MatMul`.

---

### 2.3 Explosion des Ressources : Line Buffers en Bascules (Flip-Flops) dans `im2col`

- **Fichier concerné** :
  - [`spinalML/src/spinalML/ops/im2col.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/im2col.scala#L55-L59)
- **Problématique** :
  Actuellement, les tampons de ligne de convolution sont déclarés ainsi :
  ```scala
  val lineBuffers = for (i <- 0 until K - 1) yield new Area {
    val regs = Vec(Reg(dataType), W * C)
    regs.foreach(r => r.init(r.getZero.asInstanceOf[T]))
    val pop = regs(W * C - 1)
  }
  ```
  **Ce sont de pures bascules (Flip-Flops/registres) !**
  Calcul d'impact pour une couche d'entrée d'image réaliste :
  - Soit une image de $160 \times 160$ avec $C = 32$ canaux, noyau $K = 3$.
  - Un tampon de ligne contient $W \times C = 160 \times 32 = 5\,120$ éléments de 8 bits.
  - Deux lignes ($K-1 = 2$) consomment $2 \times 5\,120 \times 8 = \mathbf{81\,920}$ **bascules** !
  - *Or*, le FPGA Sipeed Tang Primer 20K (Gowin GW2A-18) ne possède au total que **15 552 bascules** !
  - Le projet ne peut même pas synthétiser une seule couche de convolution $3 \times 3$ sur une image de plus de $32 \times 32$ pixels à cause de ce choix d'implémentation.
- **Solution & Impact** :
  - *Complexité* : **Faible / Trivial**.
  - *Solution* : Remplacer immédiatement `Vec(Reg(dataType), W * C)` par une mémoire bloc circulaire synchrone `Mem(dataType, W * C)` utilisant `readSync` et un compteur d'adresses pointeur, exactement comme c'est déjà implémenté dans [`LineBuffer2D` de `maxpool2d.scala`](file:///e:/spinalML/spinalML/src/spinalML/poolings/maxpool2d.scala#L16-L35). Les mémoires BRAM du Gowin GW2A-18 absorberont alors ces données sans consommer aucune bascule logique.
  - *Points d'impact* : `im2col.scala` uniquement (ajustement d'un cycle de latence de lecture synchrone).

---

### 2.4 Couche d'Upsampling / Interpolation Inexistante

- **Fichiers concernés** :
  - `spinalML/src/spinalML/layers/` (aucun fichier d'upsampling)
  - `spinalML/src/spinalML/ops/`
  - `spinalML/src/spinalML/nn/LayerSpec.scala`
- **Problématique** :
  Le composant essentiel des têtes de détection d'objets (FPN - Feature Pyramid Network / PANet) est l'agrandissement d'image : les cartes de caractéristiques profondes ($P5$) de faible résolution spatiale ($20 \times 20$) doivent être doublées spatialement ($40 \times 40$) pour être concaténées avec $P4$.
  Il n'existe actuellement **aucun opérateur ni couche `Upsample2D` ou `Interpolate`** dans toute la base de code SpinalML.
- **Solution & Impact** :
  - *Complexité* : **Faible à Moyenne** pour le mode *Nearest Neighbor* (le mode standard de YOLO).
  - *Solution* :
    Créer un opérateur `UpsampleNearest2D(scaleFactor = 2)` :
    1. Horizontalement : chaque pixel reçu sur le flux est répété $scaleFactor$ fois.
    2. Verticalement : chaque ligne complète est enregistrée dans une BRAM de ligne (`LineBuffer2D` de profondeur $W \times C$) et rejouée $scaleFactor$ fois avant de passer à la ligne suivante.
  - *Points d'impact* : Ajout de `Upsample2D.scala` dans `layers/`, `ops/`, exposition dans `LayerSpec.scala` et intégration dans `Sequential.scala`.

---

### 2.5 Concaténation de Tensors : Interdiction du Canal (`axis != 0`)

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/nn/LayerSpec.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/LayerSpec.scala#L249-L256)
  - [`spinalML/src/spinalML/ops/concatenate.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/concatenate.scala#L93-L116)
  - [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala#L101-L103)
- **Problématique** :
  Dans `LayerSpec.scala` :
  ```scala
  case class Concat(a: Int, b: Int, axis: Int = 0) extends LayerSpec {
    require(axis == 0, "Concat supports axis 0 only (sequential juxtaposition)")
  ```
  Le compilateur **rejette explicitement tout axe différent de 0**.
  Dans un réseau de vision (YOLO, UNet), une concaténation se fait **toujours le long de l'axe des canaux** ($C_{out} = C_A + C_B$, à dimensions spatiales $H \times W$ identiques).
  La concaténation actuelle juxtapose temporellement deux séquences entières (batch ou time-steps), ce qui est inutile pour fusionner des branches FPN ou PANet.
- **Solution & Impact** :
  - *Complexité* : **Moyenne à Élevée**.
  - *Solution* : Supporter la concaténation sur le canal (axe 2 pour la convention $[H, W, C]$ de SpinalML).
    Dans un protocole de streaming où les canaux sont sérialisés dans le temps par pixel ($lanes = 1$), la concaténation de deux flux $A$ et $B$ nécessite :
    - Soit un multiplexeur séquentiel par pixel : pour chaque pixel $(x, y)$, vider les $C_A$ valeurs du flux $A$, puis vider les $C_B$ valeurs du flux $B$.
    - Soit une concaténation en largeur vectorielle ($lanes = C_A + C_B$) via `StreamJoin`, puis repack si nécessaire.
  - *Points d'impact* : `concatenate.scala`, `LayerSpec.scala`, `Sequential.scala`.

---

### 2.6 Séparation de Canaux (`Split` / `Slice`) Non Exposée

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/nn/LayerSpec.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/LayerSpec.scala)
  - [`spinalML/src/spinalML/ops/slice.scala`](file:///e:/spinalML/spinalML/src/spinalML/ops/slice.scala)
- **Problématique** :
  Les blocs fondamentaux de YOLO modernes (CSPNet, blocs C3 de YOLOv5, blocs C2f de YOLOv8, blocs C3k2 de YOLOv11) reposent sur le découpage de la carte de caractéristiques en deux moitiés selon les canaux :
  $$X \to [X_1, X_2] \quad \text{où } X_1, X_2 \in \mathbb{R}^{H \times W \times (C/2)}$$
  Une moitié traverse une cascade de convolutions résiduelles pendant que l'autre moitié sert de raccourci (skip), avant d'être reconnectées par un `Concat`.
  Actuellement, `slice.scala` n'est pas exposé dans `LayerSpec` et ne peut pas être inséré dans un graphe `Sequential`. De plus, `SliceAxis1Op` exige que les éléments soient parallèles sur le bus (`require(end <= lanes)`), ce qui est incompatible avec les tenseurs de vision dont les canaux sont sérialisés dans le temps.
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Créer un composant `Split2D(splits = 2, axis = 2)` qui démultiplexe temporellement les canaux pour chaque pixel vers deux interfaces de sortie `Stream`.
  - *Points d'impact* : Nouveau `LayerSpec` `SplitChannel`, routage dans `Sequential.scala`.

---

### 2.7 Fonctions d'Activation Modernes : Absence de SiLU / Swish

- **Fichiers concernés** :
  - `spinalML/src/spinalML/activations/` (seuls `relu`, `leaky_relu`, `sigmoid`, `softmax`, `tanh` sont présents).
- **Problématique** :
  YOLOv5, v6, v7, v8, v9, v10, v11 utilisent universellement l'activation **SiLU (Sigmoid Linear Unit ou Swish-1)** :
  $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
  L'absence de SiLU oblige à remplacer rétroactivement toutes les activations d'un modèle pré-entraîné par des ReLU (ce qui dégrade drastiquement la précision mAP) ou à ré-entraîner le réseau de zéro.
- **Solution & Impact** :
  - *Complexité* : **Faible**.
  - *Solution* :
    - *Option A (Exacte / DSP)* : Cascadage de l'opérateur `SigmoidOp` existant avec un multiplicateur DSP matériel $x \times \sigma(x)$.
    - *Option B (Hard-SiLU / PWL)* : Approximation linéaire par morceaux (Piecewise Linear) ou ROM de 256 entrées en BRAM (identique à l'approche `MathLUTs.scala` pour les types 8 bits), ne coûtant aucun multiplicateur et s'exécutant en 1 cycle.
  - *Points d'impact* : Ajout de `silu.scala` dans `activations/`, déclaration dans `LayerSpec.scala`.

---

### 2.8 MaxPool2D : Pas de Padding pour le SPPF

- **Fichier concerné** :
  - [`spinalML/src/spinalML/poolings/maxpool2d.scala`](file:///e:/spinalML/spinalML/src/spinalML/poolings/maxpool2d.scala#L47-L48)
- **Problématique** :
  Le bloc **SPPF (Spatial Pyramid Pooling Fast)** de YOLO applique 3 maxpoolings consécutifs de taille $5 \times 5$ avec $stride = 1$ et **$padding = 2$**.
  Grâce au padding, la dimension spatiale reste strictement constante, ce qui permet de concaténer l'entrée avec les 3 sorties poolées ($C \to 4C$).
  Dans SpinalML, `MaxPool2D` calcule $H_{out} = (H - K)/stride + 1$. Pour $K = 5$, la taille diminue de 4 pixels à chaque étage, rendant impossible la concaténation SPPF.
- **Solution & Impact** :
  - *Complexité* : **Faible**.
  - *Solution* : Ajouter le support du padding dans `MaxPool2DOp` (injection de la valeur minimale lors des coordonnées virtuelles en dehors de l'image).
  - *Points d'impact* : `maxpool2d.scala`, `LayerSpec.scala`.

---

### 2.9 Convolutions Séparables et Groupées (Depthwise Conv)

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/layers/Conv2D.scala`](file:///e:/spinalML/spinalML/src/spinalML/layers/Conv2D.scala)
- **Problématique** :
  Les architectures ultra-légères (MobileNet, ShuffleNet, variantes allégées de YOLO) utilisent intensivement les convolutions par profondeur séparable (*Depthwise Separable Convolutions*) où chaque canal d'entrée est convolué indépendamment avec son propre filtre $K \times K$, sans sommation croisée entre canaux ($groups = C_{in}$).
  `Conv2D` dans SpinalML projette immédiatement l'opération sous forme de multiplication matricielle dense (`im2col` de taille $K \times K \times C_{in}$ multiplié par une matrice de poids de taille $(K \times K \times C_{in}) \times C_{out}$). Ce schéma force un produit dense complet et est incapable d'exploiter la sparsité des convolutions groupées.
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Introduire une couche `DepthwiseConv2D` dédiée qui effectue un produit scalaire spatial $K \times K$ canal par canal sans le multiplicateur matriciel GEMM complet.

---

## 3. Verrous d'Architecture Système et de Mémoire

### 3.1 Le Mur de Surface : Limite du Pipeline Déroulé Statique

- **Fichier concerné** :
  - [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala#L494-L650)
- **Problématique (Le verrou n°1)** :
  L'architecture actuelle de `Sequential` est un **pipeline spatial entièrement déroulé sur silicium** (*fully unrolled spatial streaming datapath*) :
  - Chaque couche déclarée dans `modelSpec` instancie physiquement son propre matériel sur la grille du FPGA : sa propre unité de calcul, ses propres FIFOs, ses propres contrôleurs DMA de poids et ses propres tampons.
  - Si un modèle contient 1 convolution, il instancie 1 bloc `Conv2D`.
  - Si un modèle contient **60 convolutions** (comme YOLO), SpinalML va tenter d'instancier **60 blocs matériels de convolution distincts** sur la même puce !
  
  Sur un FPGA tel que le Gowin GW2A-18 (20K LUTs, 48 DSPs) :
  - Une seule couche Conv2D avec matmul optimisé consomme environ 15 à 30% des ressources de la puce.
  - **Il est physiquement impossible de faire tenir plus de 3 ou 4 couches de convolution simultanées sur ce composant.**
  - Dérouler un réseau complet de vision sur silicium nécessiterait un FPGA industriel haut de gamme coûtant plusieurs milliers d'euros (ex: AMD Xilinx Versal / UltraScale+ avec 1 million de LUTs).

---

### 3.2 Le Mur de BRAM : `TapBuffer` et les Connexions Résiduelles Géantes

- **Fichier concerné** :
  - [`spinalML/src/spinalML/memory/TapBuffer.scala`](file:///e:/spinalML/spinalML/src/spinalML/memory/TapBuffer.scala#L39-L43)
- **Problématique** :
  Pour gérer les branches divergentes (DAG, skip connections), `TapBuffer` stocke la branche différée dans une FIFO interne sur puce :
  ```scala
  val entries = Math.max(1, depth / lanes) + 1
  val fifo = StreamFifo(Vec(dataType, lanes), entries)
  ```
  La FIFO est dimensionnée pour stocker **l'intégralité du tenseur** !
  - Dans un réseau YOLOv8, la carte d'activation $P3$ mesure $80 \times 80 \times 128$ éléments en INT8, soit **819 200 octets (~820 Ko)**.
  - La carte $P4$ mesure $40 \times 40 \times 256$ éléments, soit **409 600 octets (~410 Ko)**.
  - Le FPGA Sipeed Tang Primer 20K possède une capacité totale de BRAM d'environ **100 Ko**.
  - Une seule connexion résiduelle de YOLO fait exploser la capacité de mémoire interne du FPGA de plus de 800% !
- **Solution & Impact** :
  - *Complexité* : **Élevée**.
  - *Solution* : Soit déverser les résidus volumineux vers la RAM externe (DDR3/PSRAM), soit utiliser l'approche de streaming par bandes spatiales fusionnées ([Section 6.1.3](#613-proposition-3--z-flow--band-tiling-la-seule-vraie-réponse-mais-un-projet-de-thèse)).

---

### 3.3 Le Mur Mémoire : AXI en Lecture Seule et Absence de Contrôleur DDR3/PSRAM

- **Fichiers concernés** :
  - [`spinalML/src/spinalML/nn/Accelerator.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala#L68-L72)
  - [`spinalML/src/spinalML/io/AxiReadMem.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/AxiReadMem.scala#L25-L45)
  - [`spinalML/src/spinalML/io/UartSoC.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/UartSoC.scala)
- **Problématique** :
  1. **AXI Write désactivé** : Dans `Accelerator.scala`, les canaux d'écriture AXI sont mis à zéro :
     ```scala
     io.axiMaster.aw.valid := False
     io.axiMaster.w.valid  := False
     io.axiMaster.b.ready  := False
     ```
     L'accélérateur est strictement incapable d'écrire le moindre octet en mémoire vive externe.
  2. **Mémoire externe factice (`AxiReadMem`)** :
     Le composant `AxiReadMem` fourni par défaut est une simple BRAM interne FPGA de **32 Ko** (16 Ko pour les images, 16 Ko pour les poids).
     Les poids de YOLOv8n pèsent **3 200 Ko** (3.2 Mo) et une image $640 \times 640$ pèse **1 200 Ko** (1.2 Mo).
     La BRAM interne ne peut même pas contenir 1% du modèle.
  3. **Absence de contrôleur mémoire physique** :
     La carte Tang Primer 20K intègre une puce DDR3 de 1 Go (ou 128 Mo), mais SpinalML n'a aucun wrapper connectant l'IP Gowin DDR3 Memory Controller à son bus AXI.
- **Solution & Impact** :
  - *Complexité* : **Majeure**.
  - *Solution* :
    1. Implémenter un moteur d'écriture DMA (`DMAWriter`) sur le port AXI de l'accélérateur pour enregistrer les résultats intermédiaires et finaux.
    2. Intégrer les contrôleurs de mémoire physique (IP DDR3 de Gowin pour Tang Primer 20K, contrôleur HyperRAM/PSRAM pour d'autres cartes) pour offrir un espace d'adressage réel de plusieurs dizaines ou centaines de mégaoctets.

---

### 3.4 Topologie Multi-Têtes : Absence de Sorties Multiples

- **Fichier concerné** :
  - [`spinalML/src/spinalML/nn/Sequential.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Sequential.scala#L76-L136)
  - [`spinalML/src/spinalML/nn/Accelerator.scala`](file:///e:/spinalML/spinalML/src/spinalML/nn/Accelerator.scala#L51-L53)
- **Problématique** :
  Le composant `Sequential` et le wrapper `Accelerator` imposent une sortie unique :
  ```scala
  consumers(nNodes - 1) += -1 // -1 marks the accelerator output
  val outStream = master(Tensor(finalType, finalShape, lanes = 1))
  ```
  Tous les détecteurs de la famille YOLO (YOLOv3 à YOLOv11) possèdent **3 têtes de sortie indépendantes** associées aux 3 résolutions spatiales de la pyramide (stride 8 pour petits objets, stride 16 pour objets moyens, stride 32 pour grands objets).
  Le graphe actuel ne permet pas de déclarer plusieurs nœuds comme sorties terminales de l'accélérateur.
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Autoriser une liste de sorties `outputs: Seq[Int]` dans `Sequential` et instancier soit plusieurs canaux de streaming de sortie, soit un multiplexeur de sortie avec marquage d'en-tête de tête (`head_id`).

---

## 4. Lacunes Côté Python, CLI et Écosystème

### 4.1 Absence d'Importateur ONNX / PyTorch Automatisé

- **Problématique** :
  Actuellement, pour créer un modèle dans SpinalML, le développeur doit **écrire manuellement le modèle en Scala ligne par ligne** (ex: `examples/Mnist/Model.scala`) en recomposant à la main chaque dimension, chaque forme de poids et chaque nom de couche.
  Personne ne peut réécrire manuellement en Scala les 168 couches interconnectées d'un graphe YOLO sans introduire d'erreurs de dimension ou d'index.
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Développer un module Python `spinalml.compiler.onnx` qui prend en entrée un fichier exporté standard `yolov8n.onnx` :
    1. Parse le graphe de calcul via `onnx` ou `torch.fx`.
    2. Valide la compatibilité de chaque couche avec les primitives supportées.
    3. Génère automatiquement le code Scala `Model.scala`.

---

### 4.2 Fragilité du CLI : Parsing par Regex du Code Source Scala

- **Fichier concerné** :
  - [`cli/spinalml_cli/cli.py`](file:///e:/spinalML/cli/spinalml_cli/cli.py#L215-L255)
- **Problématique** :
  Pour détecter le composant à synthétiser et générer le lanceur automatique `AutoRunner.scala`, le CLI Python utilise des expressions régulières sur le code Scala :
  ```python
  pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
  comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
  ```
  Cette approche par regex est extrêmement fragile :
  - Si le fichier Scala contient plusieurs classes, des commentaires contenant le mot-clé `class`, ou une syntaxe d'instanciation légèrement non standard, la regex extrait le mauvais composant.
  - Le code généré est copié dans un sous-dossier temporaire `cli_temp` et recompilé avec Mill, ce qui peut provoquer des conflits de package ou des erreurs d'arborescence.
- **Solution & Impact** :
  - *Complexité* : **Faible**.
  - *Solution* : Permettre de spécifier explicitement le composant via un drapeau CLI (`--component Model`), ou utiliser un fichier manifeste de configuration clair plutôt que du scraping heuristique de code source.

---

### 4.3 Absence de Framework de Quantification Standardisé (PTQ/QAT)

- **Fichiers concernés** :
  - [`examples/Mnist/inference.py`](file:///e:/spinalML/examples/Mnist/inference.py#L69-L100)
- **Problématique** :
  Dans l'exemple MNIST actuel, la quantification a été bricolée ad-hoc dans un script d'inférence Python : conversion manuelle de flottants en entiers 4 bits empaquetés en nibbles, table personnalisée d'encodage `FP8_E4M3`, et sérialisation brute dans un `.npz`.
  Il n'existe pas d'outil clé en main dans SpinalML permettant de :
  - Prendre un modèle PyTorch entraîné en FP32.
  - Calibrer les seuils de saturation (KL-divergence ou MinMax).
  - Générer les poids quantifiés (INT8 ou W4A8) alignés sur les limites de mots AXI (64 bits).
- **Solution & Impact** :
  - *Complexité* : **Moyenne**.
  - *Solution* : Intégrer un sous-module d'exportation standardisé (utilisant PyTorch 2.x `torch.ao.quantization` ou un script de calibration PTQ autonome) produisant directement le binaire des poids `weights.bin` et la table d'alignement mémoire `MemLayout`.

---

### 4.4 Débit de la Liaison Hôte (UART à 115200 bauds)

- **Fichiers concernés** :
  - [`boards/tang-primer-20k.json`](file:///e:/spinalML/boards/tang-primer-20k.json#L7)
  - [`cli/spinalml_cli/uart_host.py`](file:///e:/spinalML/cli/spinalml_cli/uart_host.py)
- **Problématique** :
  La liaison de communication actuelle repose sur un UART série à **115 200 bauds** (soit un débit théorique maximal de ~11.5 Ko/s).
  - Pour une image MNIST ($28 \times 28 \times 1$ = 784 octets), le transfert prend ~70 ms, ce qui reste acceptable.
  - Pour une image de vision $320 \times 320 \times 3$ (~300 Ko), le transfert par UART à 115 200 bauds prendrait **26 secondes par image** !
  - Pour une image $640 \times 640 \times 3$ (~1.2 Mo), il faudrait plus de **1 minute et 45 secondes** pour envoyer une seule image !
  À ce débit, toute inférence vidéo temps réel est impossible.
- **Solution & Impact** :
  - *Complexité* : **Faible à Moyenne**.
  - *Solution* :
    1. Monter le débit UART à **2 000 000 ou 3 000 000 bauds** (supporté nativement par les puces FTDI / USB-UART des cartes Sipeed Tang).
    2. Pour du vrai temps réel : utiliser une interface hôte à haut débit (SPI haute vitesse, USB 2.0 High-Speed FT232H / FT2232H en mode synchrone FIFO, ou flux caméra DVP/MIPI-CSI direct vers le FPGA).

---

### 4.5 Découpage Détection : Post-Processing et NMS

- **Problématique** :
  Les réseaux de classification (comme MNIST) émettent simplement un vecteur de 10 scores de classes (logits).
  Un modèle de détection d'objets (YOLO) émet des milliers de boîtes englobantes candidates :
  - Par exemple sur une image $640 \times 640$, YOLOv8 émet **8 400 boîtes prédictives** contenant chacune $(cx, cy, w, h, score_{classe})$.
  - Ces prédictions doivent subir :
    1. La dé-normalisation géométrique (ancres ou distribution focale DFL).
    2. Le filtrage par seuil de confiance.
    3. Le **Non-Maximum Suppression (NMS)** pour éliminer les doublons.
  Le projet ne spécifie nulle part où doit s'effectuer cette étape.
- **Solution & Recommandation** :
  - *Complexité* : **Documentation et code hôte**.
  - *Solution* : Il est standard et recommandé de **laisser le NMS sur le processeur hôte** (Python ou microcontrôleur RISC-V), car le tri et les intersections de polygones (IoU) sont très coûteux et inefficaces en pur matériel FPGA fixe. Le matériel FPGA doit se concentrer sur l'extraction de caractéristiques et les têtes de convolution, puis renvoyer les tenseurs bruts à l'hôte.

---

### 4.6 Couverture des Tests Unitaires et de Non-Régression

- **Fichiers concernés** :
  - Dossier [`tests/python/`](file:///e:/spinalML/tests/python/)
- **Problématique** :
  La suite de tests actuelle est bien pensée (comparaison bit-exact avec des modèles de référence NumPy/PyTorch), mais elle souffre de lacunes importantes :
  - Presque tous les tests 2D s'exécutent sur des tailles miniatures arbitraires ($4 \times 4$ ou $8 \times 8$).
  - Aucun test d'intégration de chaîne 2D multi-couches complexe avec convolutions successives et résidus n'existe dans `tests/python/`.
  - Les cas limites des line-buffers (changement de ligne, bulles de flux, calage de pipeline en cours de trame) ne sont pas stressés sous forte contre-pression aléatoire.

---

## 5. Tableau Synthétique des Impacts et Priorités

| Domaine | Problème / Manque | Criticité pour YOLO | Effort Réel Estimé | Type d'Action |
| :--- | :--- | :---: | :---: | :--- |
| **RTL / Ops** | Line buffers `im2col` en bascules (LUT/FF) | 🚨 **Bloquant absolu** | Faible (1 jour) | Remplacement par `Mem` BRAM (`readSync`) |
| **RTL / Ops** | Absence de Padding (`same`, $P=1$) | 🚨 **Bloquant absolu** | Moyen (2-3 jours) | Ajout gestion bordures virtuelles dans `im2col` |
| **RTL / Ops** | Absence de Stride ($stride = 2$) | 🚨 **Bloquant absolu** | Moyen (2-3 jours) | Sous-échantillonnage temporel dans `im2col` |
| **RTL / Ops** | Couche d'Upsampling $2\times$ (Nearest) | 🚨 **Bloquant (FPN)** | Moyen (2 jours) | Nouveau composant `UpsampleNearest2D` |
| **RTL / Ops** | Concaténation le long des canaux (axis 2) | 🚨 **Bloquant (FPN)** | Élevé (3-5 jours) | Refonte multiplexage spatial/temporel dans `concatenate` |
| **RTL / Ops** | Activation SiLU / Swish | ⚠️ Majeur | Faible (1 jour) | Table ROM PWL ou $x \cdot \text{sigmoid}(x)$ |
| **RTL / Ops** | Découpage de canaux (`Split` / CSP) | ⚠️ Majeur | Moyen (2 jours) | Nouveau composant `SplitChannel` |
| **RTL / Ops** | MaxPool2D avec Padding (SPPF) | ⚠️ Majeur | Faible (1 jour) | Extension de `maxpool2d.scala` |
| **Système** | Pipeline déroulé statique (Mur de surface) | 🚨 **Bloquant pour modèles > 3 couches** | **Projet de recherche (12-18 mois)** | Synthèse dédiée repliée (Section 6, 7 & 8) |
| **Système** | FIFOs `TapBuffer` géantes pour résidus | 🚨 **Bloquant (>100 Ko)** | Élevé (3-4 mois) | Déversement externe ou halo-tiling |
| **Système** | AXI Write désactivé + BRAM 32 Ko | 🚨 **Bloquant absolu** | Élevé (1-2 mois) | Moteur AXI Write + contrôleur DDR3 physique |
| **Système** | Sorties Multi-Têtes (P3, P4, P5) | ⚠️ Majeur | Moyen (2 jours) | Support multi-sorties dans `Sequential` |
| **Python** | Importateur automatique ONNX / PyTorch | ⚠️ Majeur (Ergonomie) | Moyen (1-2 semaines) | Script de compilation de graphe ONNX |
| **Python** | Parsing regex fragile dans le CLI | 💡 Confort / Robustesse | Faible (1 jour) | Déclaration explicite `--component` |
| **Python** | Débit UART 115200 trop lent pour 640x640 | ⚠️ Majeur (Latence) | Faible (0.5 jour) | Passage à 2-3 Mbaud ou mode USB FIFO |
| **Python** | Framework de quantification propre (PTQ) | 💡 Important | Moyen (2-3 semaines) | Pipeline de calibration INT8/W4A8 unifié |

---

## 6. Démystification des Solutions de Réutilisation & Réalité Physique

Un choix d'architecture à ce niveau de refactorisation engage le projet sur plusieurs mois, voire plusieurs années. Il est crucial d'écarter tout discours séducteur pour poser un diagnostic technique froid, lucide et rigoureux.

---

### 6.1 Déconstruction Critique des 4 Approches de Réutilisation

#### 6.1.1 Proposition 1 — Folded Stream Ring (L'illusion de l'autonomie)
- **Ce que c'est réellement** : C'est le « *Layer Folding Level 2* » déjà esquissé dans la roadmap, présenté sous un habillage séduisant. C'est l'approche standard des accélérateurs FPGA pour LLM (ex: FlightLLM, architectures Kria).
- **Le piège fatal** : Le concept parle de « rebouclage » sans jamais répondre à la question physique essentielle : **où vit le flux de données pendant la recirculation ?**
  - Pour MNIST (784 octets), on peut reboucler dans une poignée de FIFOs BRAM internes.
  - Pour YOLO à $320 \times 320 \times 32$, un tenseur intermédiaire pèse **3,2 Mo**. On ne fait pas tenir 3,2 Mo dans les 100 Ko de BRAM du FPGA.
  - Dès lors, deux seules issues existent :
    1. Écrire le tenseur complet en mémoire DDR3 externe à chaque tour de boucle (ce qui sature immédiatement la bande passante mémoire et détruit la latence).
    2. Découper spatialement le calcul en bandes horizontales (*spatial tiling* $\to$ Proposition 3).
    3. Ou combiner les deux.
- **Verdict** : La proposition 1 **n'est pas autonome**. Elle est strictement dépendante du découpage spatial par bandes (Proposition 3) pour être viable sur YOLO. Complexité réelle d'ingénierie : **4 à 6 mois**, pas « moyenne ».

---

#### 6.1.2 Proposition 2 — Macro-Bloc Bottleneck Recirculant (L'écueil de la sur-spécialisation)
- **Ce que c'est réellement** : Spécialiser le silicium pour un unique bloc canonique de YOLO ($Conv_{1\times1} \to Conv_{3\times3} + \text{Shortcut}$).
- **Le problème structurel** : Un réseau comme YOLOv8 ou YOLOv11 n'utilise **absolument pas** uniquement des blocs Bottleneck. Il contient :
  - Des convolutions $1 \times 1$ de projection pures (sans résidu).
  - Des convolutions $3 \times 3$ spatiales avec $stride = 1$ et $stride = 2$ (downsampling).
  - Des convolutions *depthwise* (dans les variantes allégées nano).
  - Des blocs C2f (concaténation asynchrone de 4 branches).
  - Des blocs C3k2 (topologie variable).
  - Le bloc SPPF (3 maxpoolings consécutifs avec $padding = 2$).
  - Le couplage FPN/PANet (upsampling $2\times$ + concaténation par canal).
  - La tête de détection (*Detect head* anchor-free avec distribution focale DFL).
- **Conséquence** : Un macro-bloc Bottleneck ne couvre qu'environ **50% du réseau**. Pour traiter la moitié restante, il faudrait concevoir 4 ou 5 autres moteurs spécialisés. On aboutit alors à un pipeline déroulé hétérogène avec 6 à 8 macro-blocs complexes sur silicium, annulant tout gain d'abstraction.
- **Verdict** : Over-optimisé pour un sous-ensemble étroit de YOLO. Mauvaise abstraction architecturale. Effort estimé : **3 à 4 mois** pour un résultat fragile.

---

#### 6.1.3 Proposition 3 — Z-Flow / Band Tiling (La seule vraie réponse, mais un projet de thèse)
- **Ce que c'est réellement** : C'est la seule proposition techniquement fondamentale du document. C'est l'état de l'art mondial utilisé par les compilateurs de pointe (TVM, TensorRT) et les architectures FPGA avancées (FINN, VTA, architectures LLM custom) : **la fusion d'opérateurs combinée au partitionnement spatial en bandes (spatial tiling)**. Les activations intermédiaires ne touchent jamais la RAM externe ; elles transitent de haut en bas de l'image à travers les line-buffers locaux.
- **Ce que cela implique réellement (la réalité cachée)** :
  C'est la solution exacte au problème, mais sa complexité a été lourdement sous-estimée. Ce n'est pas un composant à ajouter, c'est un séisme complet sur le framework :
  1. **Propagation inverse du halo spatial dans tout le DAG** : Chaque convolution $K \times K$ a besoin d'une marge de contexte vertical (halo). À travers 20 couches de convolution, la taille de la bande d'entrée brute doit être calculée par rétro-propagation analytique à travers les convolutions, les strides, les concaténations et les résidus asynchrones.
  2. **Refonte complète de `Sequential`** : Tout le séquenceur matériel doit être réécrit pour accepter et émettre des tenseurs bandés à chaque étage, en gérant les temps d'arrêt de jointure (*seam stalls*).
  3. **Refonte complète de `im2col`** : Le générateur doit conserver l'état du halo entre bandes successives en prenant en compte les sauts d'échelle ($stride = 2$ divise par deux la hauteur d'une bande ; la bande suivante doit recalculer son propre décalage).
  4. **Gestion des bords géométriques** : Le padding en haut de l'image n'apparaît que sur la première bande, le padding en bas uniquement sur la dernière.
- **Verdict** : C'est la **seule bonne réponse technique**, mais c'est un travail de recherche et développement de **12 à 18 mois pour un ingénieur expérimenté**. (C'est exactement la problématique sur laquelle des projets comme TVM ont mobilisé une équipe de 20 chercheurs pendant 5 ans). Ce n'est pas un simple composant à coder.

---

#### 6.1.4 Proposition 4 — Stencil Polymorphe à ROM (Le microcode qui ne dit pas son nom)
- **Ce que c'est réellement** : Prétendre qu'une table ROM de configuration de couches n'est « ni un processeur ni du microcode » relève de la contradiction technique : **une ROM qui contient la description séquentielle des couches matérielles est par définition du microcode**. La distinction avec un NPU classique n'est que purement cosmétique : au lieu d'un jeu d'instructions assembleur, on manipule des champs de registres statiques. C'est un NPU doté d'un jeu d'instructions appauvri.
- **Le vrai problème** : L'extrême diversité des couches de YOLO. Forcer un seul composant `UnifiedStencilCore` à exécuter des convolutions $1\times1$, $3\times3$, $5\times5$, des convolutions groupées, des concaténations, de l'upsampling et du SPPF revient à fabriquer un **CGRA** (*Coarse-Grained Reconfigurable Array*). C'est notoirement l'une des architectures les plus complexes de l'électronique numérique à concevoir, cadencer et vérifier. On perd tous les avantages de SpinalHDL (dataflow typé, cycle-exact) pour retomber dans les travers d'un processeur générique sous-performant.
- **Verdict** : Un NPU déguisé, et un mauvais NPU. Effort estimé : **4 à 6 mois**.

---

### 6.2 La Vérité Architecturale : Ce Sont Tous des NPU Déguisés

Les quatre propositions sont en réalité des déclinaisons d'une même transition incontournable : **passer d'un pipeline spatial déroulé ($N$ blocs sur silicium) à un pipeline temporel (1 à 3 blocs réutilisés séquentiellement)**.

Toutes ces approches réintroduisent, à des degrés divers, les propriétés fondamentales d'un NPU :

| Proposition | Nature Réelle | Efficacité Matérielle | Effort d'Ingénierie Réel | Viabilité pour YOLO sur 20K LUTs |
| :--- | :--- | :---: | :---: | :---: |
| **1. Folded Ring** | NPU léger à boucle fermée | Bonne | **4 à 6 mois** | ⚠️ Incomplet sans Tiling (Prop 3) |
| **2. Bottleneck Macro** | NPU spécialisé ultra-étroit | Fragile (spécifique à 50% de YOLO) | **3 à 4 mois** | ❌ Insuffisant (trop de couches hors scope) |
| **3. Z-Flow (Tiling)** | NPU spatial à fusion d'opérateurs | **Excellente (Zéro DDR)** | **12 à 18 mois** | ✅ **La seule solution complète** |
| **4. Stencil ROM** | NPU à microcode statique déguisé | Médiocre | **4 à 6 mois** | ❌ Mauvais compromis |

> [!IMPORTANT]
> **Conclusion architecturale** : Sur les quatre approches, **seule la Proposition 3 (Z-Flow / Band Tiling), couplée à la Proposition 1 pour la gestion des flux de poids, est techniquement capable de faire entrer un réseau YOLO dans les 100 Ko de BRAM d'un Tang Primer 20K.**

---

### 6.3 Le Plafond Physique Infranchissable du Tang Primer 20K

Indépendamment de l'élégance du code SpinalHDL et du génie de l'architecture choisie, les limites ultimes sont dictées par la physique du composant.

#### Analyse des ressources matérielles du Gowin GW2A-18
- **Logique** : 20 736 LUT4
- **Registres** : 15 552 Bascules (Flip-Flops)
- **Multiplicateurs DSP** : 48 blocs DSP matériels (`MULT18X18`)
- **Fréquence d'horloge maximale réaliste** : ~100 MHz

#### Débit de calcul crête absolu
- 48 DSPs cadencés à 100 MHz effectuent au maximum :
  $$48 \times 100 \times 10^6 = 4.8 \text{ GMAC/s}$$
- En mobilisant une partie des LUTs pour synthétiser des MACs logiques supplémentaires (~64 MACs en LUTs), le FPGA peut atteindre au grand maximum :
  $$\text{Capacité crête totale} \approx 11 \text{ GMAC/s} \approx \mathbf{22 \text{ GFLOP/s}}$$
  *(en supposant un parallélisme parfait à 100% sans aucune bulle de pipeline ni temps mort mémoire).*

#### Confrontation aux modèles réels de la famille YOLO
En calculant le débit d'inférence maximal théorique (FPS = Capacité crête / FLOPs du modèle) :

$$\text{FPS}_{\text{théorique}} = \frac{22 \text{ GFLOP/s}}{\text{GFLOPs du modèle}}$$

1. **YOLO-Nano** (modèle expérimental ultra-compact à ~4,5 GFLOPs) :
   $$\text{FPS}_{\text{max}} \approx \frac{22}{4.5} \approx \mathbf{\sim 5 \text{ FPS}}$$
2. **YOLOv5n** (variante nano à ~7,1 GFLOPs) :
   $$\text{FPS}_{\text{max}} \approx \frac{22}{7.1} \approx \mathbf{\sim 3 \text{ FPS}}$$
3. **YOLOv8n** (variante standard nano à ~8,7 GFLOPs) :
   $$\text{FPS}_{\text{max}} \approx \frac{22}{8.7} \approx \mathbf{\sim 2.5 \text{ FPS}}$$

> [!CAUTION]
> **Le plafond de verre du silicium** :  
> Même avec la meilleure architecture au monde, une gestion de halo parfaite et un contrôleur DDR3 saturé à 100%, **un Tang Primer 20K ne dépassera jamais 2 à 3 images par seconde sur YOLOv8n**.  
> 20 736 LUT4 et 48 DSP restent physiquement 20 736 LUTs et 48 DSPs. Aucune astuce d'implémentation ne peut créer des multiplicateurs magiques sur la puce.

---

### 6.4 Diagnostic Stratégique et Arbitrage

Face à ces réalités techniques et physiques incontournables, trois voies stratégiques claires s'offrent au projet selon l'objectif poursuivi :

#### Voie 1 : L'objectif est « Faire tourner un vrai YOLO sur Tang Primer 20K, coûte que coûte »
- **Nature de la démarche** : Projet de recherche académique de pointe / démonstrateur de frugalité extrême.
- **Investissement nécessaire** : **12 à 18 mois de travail acharné** sur la fusion d'opérateurs, le tiling spatial Z-Flow et la gestion récursive de halo dans SpinalHDL.
- **Résultat attendu** : Une prouesse technique indiscutable sur la scène open-source FPGA, mais limitée physiquement à **2 ou 3 FPS**.

#### Voie 2 : L'objectif est « Faire de la détection d'objets YOLO en temps réel (30 FPS) »
- **Nature de la démarche** : Projet d'ingénierie applicative / produit fini.
- **Constat lucide** : **Le Tang Primer 20K est la mauvaise cible matérielle.**
- **Recommandation pragmatique** : Changer de carte. Migrer vers une plateforme telle que la **AMD Kria KV260** (Zynq UltraScale+ avec 256K LUTs, 1 200 DSPs et support natif du DPU Xilinx).
- **Résultat attendu** : Un modèle YOLOv8 tournant à **30 FPS temps réel** déployé en **2 semaines de travail**, au lieu de 18 mois d'effort sur une puce trop petite.

#### Voie 3 : L'objectif est « Apprendre en profondeur l'ingénierie matérielle du Machine Learning »
- **Nature de la démarche** : Terrain d'apprentissage et d'exploration pédagogique d'élite.
- **Constat lucide** : Le couple **SpinalML + Tang Primer 20K** est un environnement d'une richesse exceptionnelle pour maîtriser SpinalHDL, les bus AXI, le streaming réactif et les compromis de quantification matérielle.
- **Recommandation pragmatique** : Garder YOLO comme une **étoile polaire méthodologique** pour guider le développement de briques propres (padding, stride, line-buffers BRAM, upsampling, activations), tout en ayant une conscience absolue des frontières physiques de la puce.

---

## 7. L'Approche du Compilateur Silicium Dédié (Model-Tailored Architecture)

Cette section analyse l'approche consistant à concevoir SpinalML non pas comme un NPU monolithique générique, mais comme un **compilateur de coprocesseurs sur-mesure taillés au millimètre près pour le réseau défini par l'utilisateur**, avec repliement temporel et perspective de fabrication de puce (ASIC).

---

### 7.1 En quoi est-ce une approche conceptuellement différente ?

Dans un NPU générique commercial (comme un ARM Ethos ou un DPU), une grande partie du silicium est gaspillée dans la généralité :
- Des ALUs surdimensionnées pour supporter le FP32, FP16, INT16, INT8, INT4 au cas où.
- Des multiplexeurs d'interconnexion complexes et des crossbars pour interconnecter des blocs imprévisibles.
- Des décodeurs d'instructions et une logique de contrôle dynamique qui consomment de l'énergie et de la surface.

L'approche du **Compilateur Silicium Dédié** exploite la méta-programmation Scala de SpinalHDL pour synthétiser un circuit sur-mesure :
1. **Élagage chirurgical des bits (Zero Transistor Gaspillé)** :
   Si le modèle est quantifié en **W4A8** (poids 4 bits, activations 8 bits), SpinalHDL n'instancie ni ALU FP32, ni multiplieur $16 \times 16$. Il génère physiquement des multiplieurs câblés en **$8 \times 4$ bits signés**.
   Sur un ASIC, un multiplieur $8 \times 4$ consomme environ **4 fois moins de surface qu'un $8 \times 8$**, et **25 fois moins qu'un $16 \times 16$**.
2. **Contrôle câblé en dur (Hardwired FSM)** :
   Pas d'instructions, pas de décodeur. Une machine d'état synchrone générée par Scala applique les constantes exactes de chaque couche $(H, W, Cin, Cout, K, Stride, Pad)$ directement aux comparateurs matériels. Le contrôle ne coûte que quelques dizaines de bascules.
3. **Pruning fonctionnel complet** :
   Si le réseau utilise uniquement des convolutions $1\times1$, $3\times3$ et l'activation SiLU, le RTL généré ne contient **aucun circuit** pour du $5\times5$, aucun circuit pour du MaxPool, et aucun circuit pour du ReLU ou Tanh.

---

### 7.2 Est-ce que cette approche règle les problèmes du Point 6 ?

L'analyse factuelle et sans concession donne les résultats suivants selon les trois verrous du Point 6 :

#### 1. Le Mur de Surface (LUTs / Surface Silicium) : OUI, RÉSOLU
- En repliant les 60 couches du modèle sur un étage de calcul composite unique taillé sur-mesure (ex: un bloc convolutionnel spatial $8 \times 4$ bits), la surface logique devient constante et très faible (< 12K LUTs sur FPGA, moins de $1.5 \text{ mm}^2$ en ASIC 130nm). Le modèle tient sans problème sur la grille logique.

#### 2. Le Mur de Débit Physique (Le plafond des 2.5 FPS) :
- **Sur le FPGA Tang Primer 20K : NON, AUCUN CHANGEMENT.**
  Les 48 DSPs de la puce Gowin restent physiquement 48 DSPs. À 100 MHz, le plafond mathématique absolu pour YOLOv8n (8,7 GFLOPs) reste de **~2,5 FPS**. L'élégance du compilateur n'ajoute pas de multiplicateurs sur une puce qui n'en a pas.
- **Sur un ASIC (Fabrication de puce) : OUI, TRÈS NETTEMENT.**
  C'est ici que l'approche prend tout son sens. Sur un ASIC, on ne dépend plus des 48 DSPs figés d'un FPGA. Parce que les multiplieurs sont ultracompacts ($8 \times 4$ bits), on peut en implanter 256 ou 512 sur une surface minuscule. À 200–400 MHz sur silicium, le débit dépasse alors les **30 à 60 FPS réels**, avec une consommation inférieure à 1 Watt.

#### 3. Le Mur de Mémoire (3,2 Mo de cartes d'activation vs 100 Ko de BRAM) : NON, PAS AUTOMATIQUEMENT
- Faire « looper » un layer résout la réutilisation des calculs, **mais ne change rien au volume de données à stocker entre deux passes**.
- Si la couche 2 émet un tenseur de 3,2 Mo, il ne tient physiquement pas dans les 100 Ko de BRAM interne du FPGA.
- Pour que cette approche fonctionne, elle **doit impérativement être adossée à l'une des deux solutions de stockage** :
  - Soit un moteur d'écriture DMA vers la RAM externe (`DMAWriter` DDR3), ce qui introduit un trafic mémoire externe important.
  - Soit le découpage spatial en bandes (Z-Flow / Band Tiling), pour que le tenseur recirculant ne dépasse jamais la taille d'une bande (quelques Ko).

---

### 7.3 La structure actuelle de SpinalML a-t-elle le potentiel réel ?

Un audit objectif des composants existants de SpinalML montre un potentiel très élevé, mais avec des briques clés qui doivent impérativement être restructurées :

#### Ce qui est déjà prêt et solide (environ 65% du socle) :
1. **Le modèle déclaratif `LayerSpec`** : Le compilateur Scala parcourt déjà le graphe à l'élaboration et extrait exactement toutes les formes, types et connexions. L'analyse statique du modèle est déjà opérationnelle.
2. **Le système de double-buffering de poids (`StreamDoubleBuffer`)** : Déjà mature avec `residentHold` et `weightPrefetch`. Il est nativement conçu pour charger les poids de la couche suivante en arrière-plan pendant que la couche courante s'exécute, ce qui est parfait pour une boucle de layers.
3. **Le découplage `im2col` + `matmul`** : La séparation entre génération spatiale et multiplication matricielle est la bonne abstraction matérielle pour le hardware de vision.
4. **L'infrastructure de streaming réactif** : La gestion des flux SpinalHDL `Stream` avec poignées de main `valid`/`ready` et le vidage d'accumulateur `temporal = N` sont robustes et validés bit-exact.

#### Ce qui doit obligatoirement être refondu ou ajouté :
1. **Refonte de `Sequential.scala`** : Actuellement, `Sequential` instancie physiquement chaque couche dans une boucle `for (i <- layers.indices)`. Il faut remplacer ce déroulement spatial par **l'instanciation d'un seul cœur replié sur-mesure piloté par une FSM séquentielle d'états**.
2. **Correction des Line Buffers de `im2col.scala`** : Remplacer immédiatement les bascules (`Reg`) par de la BRAM synchrone (`Mem`), et rendre la profondeur de ligne configurable dynamiquement par compteur pour s'adapter aux différentes résolutions du modèle.
3. **Création du Scratchpad d'Activations** : SpinalML ne possède actuellement aucun composant mémoire double-banque (ping-pong) capable de stocker la sortie d'un layer pour la réinjecter à l'entrée du suivant.
4. **Activation de l'écriture AXI** : `Accelerator.scala` a ses canaux d'écriture AXI coupés (`aw.valid := False`). Un moteur `DMAWriter` est indispensable si les activations doivent transiter par la DDR.
5. **Primitives géométriques** : Implémenter le padding ($P=1$), le stride ($S=2$) et l'activation SiLU.

---

### 7.4 Synthèse et Faisabilité Factuelle

| Question | Réponse Factuelle |
| :--- | :--- |
| **Est-ce une autre approche ?** | **Oui.** C'est un compilateur silicium dédié (DSA - *Domain-Specific Accelerator*) généré sur-mesure par Scala, éliminant tout le gaspillage de surface d'un NPU générique. |
| **Règle-t-elle le problème de surface ?** | **Oui.** La surface matérielle devient minime et constante, calibrée sur le pire cas du modèle. |
| **Règle-t-elle le problème de débit sur Tang Primer 20K ?** | **Non.** 48 DSPs = 22 GFLOP/s $\implies$ 2 à 3 FPS max sur YOLOv8n. La physique du FPGA ne change pas. |
| **Règle-t-elle le problème de débit sur un futur ASIC ?** | **Oui.** En ASIC, la compacité des multiplieurs $8\times4$ permet d'en instancier des centaines et d'atteindre 30 à 60 FPS sous 1 Watt. |
| **Règle-t-elle le problème de BRAM (3,2 Mo) ?** | **Non, pas seule.** Elle doit obligatoirement être complétée par du streaming en bandes (Z-Flow) ou un moteur DMA d'écriture externe en DDR. |
| **La base de code SpinalML a-t-elle le potentiel ?** | **Oui.** Les briques fondamentales (streaming, double buffer, GEMM, LayerSpec) sont d'excellente qualité. C'est le niveau d'assemblage supérieur (`Sequential`) qui doit passer d'une guirlande déroulée à un cœur replié rebouclant. |

---

## 8. La Stratégie Dual-Target (FPGA & ASIC) : Du Sandwich Théorique au Plan Réel

Cette section clarifie définitivement la relation entre FPGA et ASIC dans SpinalML : comment concevoir le code aujourd'hui pour qu'il s'exécute sur le FPGA Tang Primer 20K tout en préparant rigoureusement une future fabrication de puce silicium (Tapeout).

---

### 8.1 Où se situe la frontière exacte entre FPGA et ASIC ?

Il ne faut pas mélanger les couches :
- **Ce qui est purement physique (et spécifique à chaque cible)** :
  1. *La mémoire interne* : Le FPGA instancie des blocs BRAM préfabriqués (EBR Gowin) ; l'ASIC utilise des macros SRAM générées sur-mesure par une fonderie (ex: compilateur **OpenRAM** pour SkyWater 130nm).
  2. *L'arbre d'horloge et le reset* : Le FPGA dispose de réseaux d'horloge globaux (BUFG) ; l'ASIC requiert une synthèse d'arbre d'horloge (CTS), un équilibrage de skew et une analyse de timing statique (STA) multi-coins (PVT).
  3. *Les interfaces I/O et PHY* : Le FPGA intègre en silicium dur des contrôleurs DDR3 et des émetteurs-récepteurs ; l'ASIC vierge nécessite l'intégration de blocs analogiques mixtes (padrings, I/O cells, PHYs).
  4. *Le coût de l'erreur* : Un bug sur FPGA se corrige en 2 minutes par reflash ; un bug envoyé en fonderie entraîne une puce non fonctionnelle et la perte de plusieurs dizaines de milliers d'euros.
- **Ce qui est 100% commun et directement transférable** :
  Le dataflow en streaming réactif (`Stream[Tensor]`), les machines d'états (FSM), les compteurs, la logique de repliement temporel, les arbres d'addition, et les multiplicateurs arithmétiques. Tout cela se synthétise identiquement en portes logiques standard avec Yosys/OpenROAD sur ASIC ou avec Yosys/nextpnr sur FPGA.

> [!NOTE]
> **Le FPGA n'est pas un détour, c'est l'émulateur matériel obligatoire de l'ASIC.**  
> Dans l'industrie des semi-conducteurs, aucune puce complexe n'est envoyée en fonderie sans avoir été préalablement validée de manière exhaustive sur FPGA. Développer pour le FPGA n'est donc pas une perte de temps : c'est l'étape de vérification bit-exact indispensable.

---

### 8.2 L'Architecture Sandwich à 3 Couches

Pour que le code source SpinalHDL serve simultanément les deux cibles sans réécriture, l'architecture doit s'organiser selon trois niveaux d'abstraction stricts :

```
┌────────────────────────────────────────────────────────────────────────┐
│  COUCHE 1 : CŒUR ALGORITHMIQUE SPINALML (100% Portable FPGA & ASIC)     │
│  - Modèle déclaratif (LayerSpec, DagDSL)                               │
│  - Moteur replié (Folded Core FSM) & boucle de couches                 │
│  - Pipeline de streaming (im2col, GEMM, SiLU, Upsample, Add, Concat)   │
│  - Bus internes neutres (Axi4Stream / Stream[Tensor])                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  Interface mémoire abstraite
┌───────────────────────────────────▼────────────────────────────────────┐
│  COUCHE 2 : ADAPTATEUR MÉMOIRE (La seule couche qui permute)          │
│  - Cible FPGA : Wrapper vers BRAM Gowin ou IP DDR3 Gowin               │
│  - Cible ASIC : Wrapper vers macros OpenRAM SRAM ou contrôleur externe │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  Broches physiques
┌───────────────────────────────────▼────────────────────────────────────┐
│  COUCHE 3 : PHYSIQUE (Puces & Boîtier)                                 │
│  - Cible FPGA : Contraintes de broches .cst (Tang Primer 20K)          │
│  - Cible ASIC : Padring, Leadframe, contraintes de placement OpenROAD  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 8.3 L'Audit Réel : Pourquoi le code actuel est à ~20% et non à 80%

Le concept du « sandwich » est la bonne direction théorique, mais un audit lucide de la base de code actuelle montre que l'isolation n'est pas encore effective :

1. **`lanes` paramétrable partout** :  
   *État* : Partiellement réalisé. Déjà intégré dans `Linear.weightLanes` et `Conv2D.outLanes`, mais de nombreux modules de tuyauterie interne supposent encore `lanes = 1` en dur.
2. **Mémoire synchrone universelle (`Mem.readSync`)** :  
   *État* : **Non fait dans `im2col.scala`**. L'opérateur utilise encore un vecteur de registres (`Vec(Reg)`), ce qui empêche le mapping aussi bien en BRAM FPGA qu'en macro SRAM ASIC.
3. **Multiplieurs / DSP abstraits** :  
   *État* : À moitié fait. `DspConfig` et `DspTarget` existent, mais [`Conv2DLayer.scala`](file:///e:/spinalML/spinalML/src/spinalML/layers/Conv2D.scala) appelle directement `DspMul`. Il manque un trait unifié `Target` permettant de choisir explicitement entre `Target.FPGA` (inférence DSP ou primitives Gowin) et `Target.ASIC` (arbre de cellules standard sans primitives propriétaires).
4. **Séparation Cœur / Adaptateur mémoire** :  
   *État* : **Non fait**. Le composant [`AxiReadMem.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/AxiReadMem.scala) est directement codé en dur dans le top-level [`UartSoC.scala`](file:///e:/spinalML/spinalML/src/spinalML/io/UartSoC.scala), liant l'accélérateur à une BRAM spécifique au lieu d'une interface abstraite `MemoryAdapter`.

---

### 8.4 La Méthode « Faire les Deux » en 4 Étapes Séquencées

Vouloir servir le FPGA et l'ASIC ne signifie pas mener deux projets de front, mais **ordonner rigoureusement un projet unique en quatre étapes chronologiques** :

* **Étape 1 (0 → 3 mois) : Rendre le socle rigoureusement portable**
  - Refactoriser `im2col.scala` pour utiliser `Mem.readSync`.
  - Abstraire `DspMul` derrière un trait `Target` (`Target.FPGA` $\implies$ mapping DSP Gowin/Xilinx ; `Target.ASIC` $\implies$ synthèse logique pure `a * b`).
  - Isoler `AxiReadMem` derrière une interface `MemoryAdapter` (`BramAdapter`, `DdrAdapter`, `SramAsicAdapter`).
  - Généraliser le paramètre `lanes` sur l'ensemble de la chaîne de flux.
  - *Livrable* : Le code compile et simule bit-exact pour FPGA (Tang Primer) et pour ASIC (cellules standard en simulation Verilator).

* **Étape 2 (3 → 9 mois) : Implémenter les primitives de vision pour les deux cibles**
  - Développer le padding virtuel (`same`), le sous-échantillonnage temporel ($stride = 2$), l'activation SiLU (ROM PWL), l'upsampling $2\times$ et la concaténation de canaux (axis 2).
  - Implémenter le tampon double-banque d'activations en mémoire synchrone (compatible BRAM et OpenRAM).
  - *Livrable* : Le framework sait décrire et simuler un CNN complet représentatif de YOLO.

* **Étape 3 (9 → 18 mois) : Le Cœur Replié (Folded Core) et le Séquenceur Temporel**
  - Refondre `Sequential.scala` en un moteur d'exécution temporel piloté par une FSM matérielle de descripteurs de couches.
  - Valider l'exécution couche par couche sur silicium FPGA réel (Tang Primer 20K).
  - *Livrable* : Un coprocesseur ML compact, fonctionnel sur FPGA et prêt pour le placement-routage ASIC.

* **Étape 4 (18 → 30 mois) : Industrialisation et Tapeout Silicium**
  - Intégrer le flux OpenLane / OpenROAD pour SkyWater 130nm (SKY130) ou GlobalFoundries 180nm (GF180MCU).
  - Participer à une navette de prototypage (ex: Tiny Tapeout ou shuttle multi-projet MPW).
  - *Livrable* : Réception et mise en banc de test de la première puce physique.

---

### 8.5 Ne pas se tromper de combat : Les décisions à coût zéro aujourd'hui qui sauvent 6 mois demain

Il ne faut pas se tromper de combat. La question n'est pas de choisir entre FPGA ou ASIC, ni de courir immédiatement après l'implémentation de 15 nouveaux layers de vision alors que les fondations matérielles ne sont pas encore portables.

#### Le piège de la dette technique différée
Si on ajoute aujourd'hui des convolutions stridées, du SiLU ou des blocs résiduels sur la base actuelle (avec des `Vec(Reg)` dans `im2col`, des appels directs à `DspMul`, et un `AxiReadMem` soudé dans le SoC), chaque nouveau layer multipliera le couplage matériel :
- Dans 6 mois, lorsqu'il faudra compiler pour OpenLane ou cibler une autre famille FPGA (Xilinx, ECP5), il n'y aura pas 3 fichiers à retoucher, mais **plus de 200 sites de code à défaire et réécrire**.
- À l'inverse, poser les abstractions aujourd'hui (`Mem.readSync`, trait `Target`, interface `MemoryAdapter`, `lanes` générique) coûte **quasiment zéro jour d'effort supplémentaire**, car le volume de code existant est encore compact et parfaitement maîtrisable.

#### La règle d'or : Refactoriser le socle avant d'étendre
Pour que SpinalML devienne réellement *FPGA and ASIC friendly*, la priorité absolue n'est donc pas d'ajouter des fonctionnalités « vitrines », mais de **purger la dette d'abstraction du socle existant** (passer des 20% actuels aux 100% de la Couche 1 et Couche 2).

> [!IMPORTANT]
> Le plan d'exécution exhaustif, module par module, de ces refactorings structurants est désormais consigné dans la [Section 10 de la ROADMAP](file:///e:/spinalML/docs/roadmap.md#10-dual-target-fpga--asic-roadmap-sequenced-refactoring-plan).

