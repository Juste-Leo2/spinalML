# Plan : reverse-engineering des blocs DDR manquants (DIFFÉRÉ)

> **Statut : À PLUS TARD.** Ne commencer qu'après silicium vert via Gowin EDA
> (cf. `docs/plan-eda.md` §4 : bitstream qui marche + `gowin_unpack`).
> **But** : implémenter le support `OSER4_MEM` / `IDES4_MEM` / `DQS` / `DLL`
> dans nextpnr-himbaechel et l'envoyer en PR upstream, pour débloquer la
> voie 100% open-source. Gap détaillé : `docs/nextpnr-ddr-gap.md`.
> Issue : `docs/issuePNR.md` (poster d'abord, travailler ensuite).

## 0. Principe (pourquoi c'est faisable vite avec LLM en appui)

Le travail n'est PAS du fuzzing aveugle : on dispose (après l'étape EDA)
d'un bitstream **de référence qui marche** sur nos pins exacts. `gowin_unpack`
(apicula) le désassemble en fuses → on **lit** comment l'EDA configure les
blocs IOLOGIC/DQS/DLL pour notre cas → on encode ces choix dans le packer
nextpnr. Le code à toucher est circonscrit : `pack_iologic.cc`
(+ listes `gowin.h:51-61`, bucket `gowin.cc:1071`). Le LLM accélère
l'écriture du C++, pas la compréhension HW — d'où l'ordre : silicium
d'abord, code ensuite.

## 1. Prérequis (tous obligatoires avant de coder)

1. [ ] Silicium vert via EDA (`docs/plan-eda.md` §3 : inférence bit-exact
   sur modèle > BRAM).
2. [ ] Bitstream `.fs` de référence + version EDA notée.
3. [ ] `gowin_unpack -d GW2A-18 -o ref.v ref.fs` OK : repérer les fuses des
   IOLOGIC de nos pins DQ/DM/DQS + config DQS/DLL.
4. [ ] Issue upstream postée (`docs/issuePNR.md` §3 cochée) — éviter le
   doublon et capter les conseils des mainteneurs (gatecat et co. sont
   actifs sur le Gowin en ce moment).
5. [ ] Machine de build : cmake récent, Boost, Eigen3, Python + apycula
   (cf. README nextpnr : `cmake .. -DARCH=himbaechel -DHIMBAECHEL_UARCH=gowin`).

## 2. Méthode (dans l'ordre)

1. **Lire la référence** : dans `ref.v` désassemblé, noter pour chaque pin
   DQ/DM/DQS : mode IOLOGIC (fuses), routage DQS (quel DQSBUFM, quelle bank),
   valeur STEP/LOCK de la DLL, contraintes HCLK. Documenter le tout dans ce
   fichier (section 5) avant d'écrire une ligne de C++.
2. **Comparer à la DB apicula** : les modes IOLOGIC MEM existent-ils déjà
   dans `apycula` (`IOLOGICA/B.modes`) ? Si oui, le travail nextpnr = pur
   packing (cas favorable). Sinon, noter les fuses manquantes (cas long :
   impliquer apicula aussi).
3. **Implémenter, dans l'ordre** (un type à la fois, testé à chaque fois) :
   a. `OSER4_MEM` (+ `IDES4_MEM`, symétrique) dans `pack_iologic()`
      + listes `type_is_iologic{o,i}` + bucket.
   b. `DQS` (bind BEL adjacent à l'IOB contraint, modèle `pack_dlldly`).
   c. `DLL` (bind + STEP/LOCK vers le pack existant).
   Règle : mimer les fonctions existantes (`pack_bi_output_iol`,
   `pack_ides_iol`, `pack_dlldly`), ne rien réinventer.
4. **Valider avec NOTRE design** (meilleur test au monde pour ce patch) :
   rebuild nextpnr → `build … --dram` complet (synth + PnR + `gowin_pack`) →
   flash → inférence bit-exact. Le design qui a révélé le bug devient la
   non-régression du patch.
5. **PR upstream** : diff minimal + cas de test + résultats silicium. Suivre
   la review (clang-format exigé pour le dev nextpnr).

## 3. Garde-fous (pour aller vite sans tout casser)

- Ne toucher QUE les 4 types DDR (pas de refacto du packer).
- Tester chaque type isolément d'abord (sonde 15 lignes, cf.
  `docs/nextpnr-ddr-gap.md` §6) avant le design complet.
- Garder un nextpnr stock à côté pour la non-régression (nos builds
  non-DRAM doivent rester verts).
- Si le chantier dépasse ~2-3 semaines de fond : re-évaluer (le silicium
  via EDA reste acquis, rien n'est perdu).

## 4. Risques

- Les fuses MEM-mode peuvent manquer côté apicula aussi (alors le chantier
  déborde sur apicula : fuzzing, long).
- Review upstream imprévisible (délai, exigences). Le patch local reste
  utilisable en attendant (binaire custom, cf. §5 quand il existera).
- Ne JAMAIS intégrer un nextpnr custom dans `spinalml setup` par défaut :
  la toolchain pinnée reste oss-cad-suite (reproductibilité).

## 5. Lecture de la référence EDA (à remplir après §1)

- IOLOGIC mode des pins DQ : ____
- IOLOGIC mode des pins DM/DQS : ____
- Config DQSBUFM (bank, groupe) : ____
- STEP/LOCK DLL observés : ____
- Modes déjà dans apicula ? : ____
