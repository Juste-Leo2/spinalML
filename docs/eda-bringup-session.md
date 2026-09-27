# Session silicium EDA — 26/09/2026 : DramSoCTop sous Gowin Designer

> **Objectif** : bitstream `DramSoCTop` sur Tang Primer 20K via les outils
> vendeur (contournement du gap nextpnr `_MEM`, cf. `docs/nextpnr-ddr-gap.md`).
> **État en fin de session** : problème isolé — GowinSynthesis vaporise le
> `litedram_core` généré par Migen (31 LUT) alors que Yosys garde ~3k LUT sur
> les mêmes fichiers et que tout le Verilog Spinal passe. Pas de bitstream.
> **Contexte** : `docs/plan-eda.md` (plan), `docs/liteDRAM.md` (journal DRAM).

## 0. Environnement

- Gowin FPGA Designer **V1.9.11.03 Education**, natif **Windows**.
- Projet : `E:\GOWIN-PROJECT\DRAM\DRAM.gprj`, device `GW2A-LV18PG256C8/I7`
  (Device Version C), top `DramSoCTop`.
- Sources (copiées de `E:\spinalML`) dans `E:\GOWIN-PROJECT\DRAM\src\`.

## 1. Création projet + enregistrement des fichiers

- Le wizard crée un template `src/add.v` ; la GUI ne rescanne pas le dossier
  seule. Fix appliqué direct dans `DRAM.gprj` (`FileList`, `enable=1`) +
  `impl/temp/rtl_parser_arg.json` (régénéré depuis le `.gprj` de toute façon,
  avec `"Top" : "DramSoCTop"` qui était vide) :
  - `src/DramSoCTop.v` + `src/litedram_core.v` (+ `src/pins.cst`,
    `type="file.cst"`, vérifié identique à `hw_build/.../pins.cst`).
- Top module : pas de menu dans l'onglet Design → passer par l'onglet
  **Hierarchy** → clic droit `DramSoCTop` → *Set As Top Module*.
- `pins.cst` : syntaxe `IO_LOC`/`IO_PORT` per-bit + `SSTL15/SSTL15D`
  compatible EDA (à charger avant le PnR ; la synthèse s'en passe).

## 2. Piège des modules dupliqués (EX3794)

`DramSoCTop.v` (519 Ko, **38 modules**) est auto-suffisant : il contient déjà
`UniversalScaleDemo`, `Sequential`, `UartBridge/Rx/Tx`, etc. Ajouter aussi les
fichiers séparés du `rtl/` (`UniversalScaleDemo.v`, `Uart*.v`, `AxiReadMem.v`)
duplique tout → `EX3794 Duplicate module name` en cascade. (Le flow Yosys ne
voit rien grâce à `filter_unique_verilog_files`.) `AxiReadMem.v` n'est ni
défini ni référencé dans ce design DRAM : inutile. Projet final : 2 fichiers.

## 3. Fix rPLL DEVICE (EX0206) — FAIT et vérifié

- Synthèse 1 : `EX0206 parameter "DEVICE" value invalid` sur l'instance
  `rPLL`, émise avec `.DEVICE("GW2A-LV18PG256C8/I7")` (nom complet).
- Cause : LiteX `GW1NPLL.do_finalize` fait `p_DEVICE = devicename` verbatim ;
  notre yml mettait le part number complet dans `devicename`.
- Fix (`dram/configs/tang-primer-20k.yml`) : `devicename: "GW2A-18C"`
  (modèle + révision C, comme le `.gprj`). Cette clé ne sert qu'au PLL
  (pinout/fréquences inchangés). Régénéré via `dram-gen`
  (`out/dram-gen-fix.log`), `.DEVICE("GW2A-18C")` confirmé l.11862,
  recopié dans `GOWIN-PROJECT/src/`.
- EX0206 a disparu. Validé par la réf externe (voir §6).

## 4. Le sweep : symptômes

- Synthèse « verte » (zéro erreur) mais rapport ridicule : **32 REG, 15 ALU,
  31 LUT** au lieu de ~8,9k LUT (Yosys, `out/scale-build.log`).
- `NL0002 ... swept in optimizing` : 39 instances en Verilog-2001
  (`soc_acc` en premier, puis Sequential/DMA/UART/Bridge/DdrAdapter…),
  **zéro warning en SystemVerilog 2017** pour le même résultat —
  en SV l'absence de warnings ne veut rien dire, seuls les chiffres comptent.
- Warnings restants, tous bénins : EX3780 (Migen non-assignés → init, normal),
  EX3791 (troncatures d'expressions, connues LiteDRAM), EX3670 (largeurs
  `RCLKSEL`/`w_param_id`), EX1998 (sideband AXI non drivés), CV00xx
  (champs AXI inutilisés).
- Horloges VIVANTES au rapport : `clk` 27 MHz (Fmax 276 !), rPLL 54 MHz,
  CLKDIV, chemin `multireg → DQS` temporisé. Pas d'erreur nulle part.
- Netlist `.vg` chiffrée (Education) : pas de forensique possible.

## 5. Bissection (décisif)

| Top | Résultat | Verdict |
|---|---|---|
| `UartTx` (Spinal) | OK (~20 regs) | style Spinal de base digéré |
| `UniversalScaleDemo` (Spinal, 32 mods) | OK (~7k LUT) | tout le Verilog Spinal passe |
| `LiteDramAxiBridge` | ~0 | **artefact** : pads DDR enterrés, sweep attendu, ne prouve rien |
| **`litedram_core` (vrais pads)** | **32 REG / 31 LUT** | **le core Migen s'effondre sous GowinSynthesis** |

Réf Yosys : `litedram_core` seul ≈ 3k LUT (P1). Donc mêmes entrées,
Yosys garde, Gowin jette. Le problème est circonscrit au Verilog généré
par Migen face à GowinSynthesis (pas le Spinal, pas l'intégration, pas
les contraintes).

## 6. Référence externe (rassurant)

`nand2mario/ddr3-tang-primer-20k` (contrôleur DDR3 qui tourne sur cette
carte exacte en EDA, 1377 LE, `/tmp/opencode/ddr3-ref`) :
- `DEVICE = "GW2A-18C"` → identique à notre fix ✓
- pins DDR (`A[10]=K3, A[9]=F9…`) → identiques aux nôtres ✓
- projet en **SystemVerilog** (config prouvée ; on s'est alignés),
  `PULL_MODE=NONE DRIVE=8 BANK_VCCIO=1.5` sur les pins DDR (les nôtres sont
  plus frustes : `SSTL15` seul — à enrichir si silicium capricieux).
- Eux utilisent `OSER8_MEM` (DDR3-800) ; nous `OSER4_MEM` (DDR3-54).

## 7. Pistes écartées (avec raison)

- **EX3780 en cascade** : les signaux existent aussi dans les designs qui
  marchent (Migen normal) ; Yosys les constante pareil et garde 8873 LUT.
- **Horloge morte** : `clk` détectée + Fmax 276 MHz + chemins `por_count`
  et `multireg→DQS` temporisés.
- **Cercle reset↔init_done** : `reset = por || !resetN || !initSync`
  (`DramSoCTop.scala:96`, `init_done` = compteurs purs du
  `DramInitSequencer`, pas d'attente externe). En synthèse il n'y a pas de
  notion de « bloqué pour toujours », seulement de constance structurelle :
  `init_done` est dynamique → ne peut pas geler la synthèse. (Le cercle
  reste un point de vigilance *silicium*, pas la cause du sweep.)
- **Constantes Yosys vs Gowin** : mêmes entrées, conclusions opposées →
  divergence de l'outil, pas du design.

## 8. Prochaines étapes

1. Creuser le couple Migen×GowinSynthesis : options de synthèse EDA
   (désactiver l'optimisation inter-modules ? keep-hierarchy ?), synthèse
   CLI `gw_synthesis --help`, ou découper `litedram_core.v` par sous-module
   pour trouver le premier qui s'effondre.
2. Piste de contournement : synthèse tierce (le projet accepte
   SynplifyPro en `.vm`) si GowinSynthesis reste bloqué sur le style Migen.
3. Fallback PHY : le contrôleur nand2mario (prouvé silicium) derrière notre
   `DdrAdapter` si LiteDRAM ne passe jamais l'EDA.
4. Une fois la synthèse pleine (~9k LUT) : `.sdc` 27 MHz → PnR (le vrai
   test `OSER4_MEM`/`DQS`/`DLL` côté vendeur) → `.fs` → flash
   (`openFPGALoader -b tangprimer20k`, natif Windows, pas d'usbipd) →
   validation UART bit-exact vs `ModelReplica`.
5. Doc : consigner la version EDA exacte + archiver le `.fs` qui marche +
   `gowin_unpack` pour le dossier nextpnr (`docs/plan-reverse.md`).

## 10. Voie B : dissection par sous-blocs (27/09, dans la foulée)

- Migen aplatit tout (`verilog.convert` : même un parent/enfant trivial →
  1 module). Donc `dram/gen/hier_subblocks.py` : `convert` fragment par
  fragment (`list_signals` + `list_special_ios` comme ports) →
  `dram/out/hier/sub_{crg,ddrphy,core,initseq,axi2native}.v` →
  `GOWIN-PROJECT/src/` + `.gprj`. Leçon : lister les signaux par attributs
  rate les signaux statement-locaux (ex. `ext_dfi` d'initseq) ; le fragment
  est la source vérité (`Module.get_fragment()` = usage unique !).
- Résultats UI (Verilog-2001, CST désactivé pour les runs partiels) :
  `sub_crg` 25 logic/20 reg, `sub_initseq` 68 LUT/24 reg,
  `sub_axi2native` 696 logic/268 reg/10 BSRAM, **`sub_core` 3549 logic /
  1462 reg** — tout est SAIN sauf **`sub_ddrphy` : erreur dure `CK0021 :
  OSER4_MEM cannot drive OBUF (dqs_o)`**, pas de rapport.
- Racine : le `GW2DDRPHY` Migen drive chaque paire DQS via `ELVDS_IOBUF`
  (2 instances), que GowinSynthesis abaisse en OBUF single-ended —
  topologie illégale après un OSER. La réf prouvée-silicium fait
  `assign DDR3_DQS = oen ? 1'bz : buf;` + `SSTL15D` en paire sur un seul
  port logique (`tang20k.cst:91-94`).
- Fix (dans le flow, pas dans le venv) : `gowin_gen.py` post-passe
  `_patch_elvds_iobuf` (2 instances → tristate + lecture ref-style),
  `dqs_n` sorti des `ios` (N généré par la tuile SSTL15D), `pins.cst`
  (2 copies) : lignes `dqs_n` supprimées, `dqs_p` seul en SSTL15D.
  Regen via `cli/main.py dram-gen`, Yosys `read_verilog + hierarchy
  -check` vert, recopié EDA.
- À valider : top=`litedram_core` → ~4k LUT attendus, zéro CK0021.
- Collatéral : `PA2122` (BSRAM WRITE_MODE DPB, étape PnR sur axi2native),
  `PA2024` (port-count, artefact du montage standalone), à traiter après.

## 11. Cure syn_keep + headless (27/09 après-midi)

- SUG550 §5.8 : `/* synthesis syn_keep=1 */` supporté. Test headless
  (`GowinSynthesis.exe -prj`, `.prj` calqué sur `impl/gwsynthesis/DRAM.prj`,
  runs dans `out/gsruns/`) : base = 32 reg/31 LUT reproduit hors GUI ;
  avec keeps sur les 2357 nets internes → **1915 reg / 3588 LUT / 5 BSRAM /
  53 SSRAM, zéro erreur**. L'optimizer repliait du vivant ; les keeps le
  bloquent. Échelle cohérente avec Yosys (~3k LUT).
- Fix intégré au flow (`gowin_gen.py` : `_apply_syn_keep` default-on après
  `_patch_elvds_iobuf`, regen via `cli/main.py dram-gen`, 516 KiB), ignoré
  par Yosys/nextpnr. Minimisation du keep-set = polish ultérieur.
- `io_resetN` T10 → **T2** (2 copies CST, pin du bouton ref nand2mario).
- Clock report du run 31 LUT : rPLL/CLKOUT* + CLKDIV présents, mais Fmax
  seulement sur `clk27` (276 MHz) et CLKDIV — que le résidu était horlogé.
- Pistes abandonnées (avec raison) : wrapper hiérarchique (namespaces par
  fragment divergents : `por_clk_1` ≠ `por_clk` entre sub-files, bridges
  infidèles), scan multi-drivers textuel (faux positifs Migen), keeps
  ciblés horloges/resets (un keep ne rend pas un constant dynamique).
- Prochaine étape : run complet top=`DramSoCTop` (théorie : le cascade
  NL0002 venait du core mort qui tenait le reset ; core vivant → SoC vivant).

## 9. Logs et pièces

- `E:\spinalML\out\scale-*.log` (sim scale bit-exact 30350 cyc, build Yosys
  8873 LUT, `dram-gen-fix.log`), `docs/bugs/2026-09-spill-*-session.md`
  (bug reArm/fire fixé juste avant, scale débloquée).
- `E:\GOWIN-PROJECT\DRAM\impl\gwsynthesis\DRAM.log`,
  `DRAM_syn.rpt.html`, `DRAM_syn_rsc.xml` (31 LUT).
- `E:\GOWIN-PROJECT\DRAM\src\` : les 2 `.v` + `pins.cst`.
