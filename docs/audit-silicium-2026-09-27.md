# Audit silicium Tang Primer 20K — 27/09/2026

> Document d'état pour commit + compaction. Journal détaillé au jour le jour :
> `docs/eda-bringup-session.md` (§0–15). Bug spill traité à part :
> `docs/bugs/2026-09-spill-bias-rearm-race-session.md`.

## 0. État en une page

- **MNIST (UartSoC, sans DRAM) : MARCHE sur silicium, des deux flows.**
  `examples/Mnist/top.fs` (précompilé) et `hw_build/mnist-control/top.fs`
  (rebuild OSS frais) répondent au ping UART sur COM8, upload poids,
  inférence OK, HW==NumPy (non-bit-exact attendu, effet RNE documenté).
- **DRAM (DramSoCTop, UniversalScaleDemo) : PLACÉE mais MUETTE.**
  Synthèse vendeur 8360 LUT / 4765 reg / 43 BSRAM, PnR PASS
  (16 IDES4_MEM + 20 OSER4_MEM + 25 OSER4 + 25 IODELAY, DLL 1/4, DQS 2/9),
  `.fs` 7093 KiB flashé OK — mais aucun port UART ne répond au ping `V`.
- **Chaîne headless complète : OPÉRATIONNELLE** (`scripts/eda_flow.py`,
  presets `dram`/`mnist`) : `compile --dram` → `gw_sh` (syn+PnR+bit) →
  verdict PASS/FAIL → flash `programmer_cli` → validation. Zéro GUI.
- **Base de comparaison : ACQUISE.** Même RTL MNIST par les deux flows
  (chiffres §5). Le silence DRAM est isolé : ni le flow, ni les pins,
  ni l'USB-outil, ni l'UART (tous prouvés par le contrôle MNIST).
- **Suspect n°1 restant : reset tenu par `init_done=0`**
  (init JEDEC jamais finie) ou `pll_locked=0` (rPLL ne locke pas).
  GAO bloqué (lecture JTAG pend, même avec trigger à 0).

## 1. Matériel et environnement

- Carte : Sipeed Tang Primer 20K, `GW2A-LV18PG256C8/I7` (GW2A-18, rev C).
- DDR3 : H5TC1G63EFR 128 MiB, sys 27 MHz / DDR 54 MHz (ratio 1:2, DLL off).
- Gowin FPGA Designer **V1.9.11.03 Education**, natif **Windows**.
- Host : WSL (lecture/édition) + Windows (toute exécution) via `powershell.exe`.

## 2. Méthodologie commandes Windows

```powershell
# Environnement CLI (typer/rich, Mill, Verilator) :
.\.venv\Scripts\python.exe cli\main.py <commande> ...

# Python géré migen/litex/litedram (NE PAS confondre avec .venv) :
C:\Users\leona\.spinalml_tools\.venv\Scripts\python.exe ...

# Yosys OSS (PATH à préfixer, sinon DLL manquantes / exit -1073741515) :
$env:PATH = 'C:\Users\leona\.spinalml_tools\oss-cad-suite\bin;' + $env:PATH

# Synthèse Gowin headless (prouvée, équivalente GUI) :
C:\Gowin\Gowin_V1.9.11.03_Education_x64\IDE\bin\GowinSynthesis.exe -prj foo.prj

# PnR + bitstream headless (prouvé) : voir scripts/eda_flow.py
# Flash vendeur CLI (driver FTDIBUS, PAS de bascule Zadig) :
C:\Gowin\...\Programmer\bin\programmer_cli.exe --device GW2A-18C --fsFile foo.fs
# Flash open source (exige WinUSB sur Interface 0 via Zadig) :
.\.venv\Scripts\python.exe cli\main.py flash <fichier.fs>
```

- **Règle drivers** : Interface 0 (JTAG) = WinUSB pour openFPGALoader,
  FTDIBUS pour outils Gowin (Programmer, GAO). Interface 1 (UART/COM)
  = ne jamais toucher. Zadig ne sait PAS réinstaller FTDIBUS
  (restauration : Gestionnaire de périphériques → désinstaller →
  rebrancher, ou réinstaller le CDM FTDI).
- **Câble** : `USB Cable Setting` à re-pointer après chaque déplacement
  (location 289/290 ↔ 0). Câble USB-C **DATA** exigé ; symptômes USB
  marginale : descripteurs illisibles, `bulk write failed`, GAO qui pend.
- **UART** : 115200 8N1, COM8 sur cette machine (auto-scan dans les scripts).

## 3. Chaîne de génération (reproductible)

```powershell
# 1. RTL Spinal + core LiteDRAM (une commande fait les deux) :
.\.venv\Scripts\python.exe cli\main.py compile tests\universal\UniversalScaleDemo.scala --dram
#   -> rtl/DramSoCTop.v (+ chain) + dram/out/litedram_core.v (patch ELVDS + keeps inclus)

# 2. Factory complète (compile + stage + gw_sh + verdict + flash/validate) :
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset dram [--flash] [--validate]
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset mnist --workdir E:\eda-factory\MNIST --skip-compile
#   -> E:/eda-factory/<PRESET>/{src,build.tcl,impl/{gwsynthesis,pnr}}
```

- `dram/gen/gowin_gen.py` : CRG maison (GW2APLL + CLKDIV, pas de DHCEN),
  `DramInitSequencer` (init JEDEC gateware, remplace le BIOS),
  `_patch_elvds_iobuf` (CK0021), `_apply_syn_keep` (sweep, §4).
- `boards/constraints/tang-primer-20k.cst` (SOURCE, à jour) : reset **M10**
  (bank 2, GPIO sans CFG, pull-up, reset virtuel — POR fait le job),
  `dqs_p` seul en SSTL15D (N auto-généré, prouvé au PnR : `G3=dqs_p[0](n)`).
  Historique pins : T10 = SSPI (PR2017), T2 = bank 4 1.5V (CT1136),
  A6 = TDI JTAG. Plus aucun LVCMOS en banks 4/5/6 (banks DDR 1.5V ;
  décodées depuis `PBGA256.json`, vérif champ `CFG` dual-purpose).
- `examples/UniversalScale/silicon_validate.py` : ping `V`, upload
  (1024B input + 65600B poids slice-transposés + bias), CSR, trigger,
  lecture 64B, oracle NumPy **vérifié bit-exact vs sim**
  (first3 `[7,7,-42]`, last3 `[-25,-46,10]`).

## 4. Bugs rencontrés (statut)

| # | Symptôme | Cause | Fix | Statut |
|---|---|---|---|---|
| 1 | Hang spill multi-beats | Course `reArm`/`fire` BiasAddOp | `ready := !reArm` + bulles | **FIXÉ**, sim bit-exacte 30350 cyc |
| 2 | `EX0206 DEVICE invalid` (rPLL) | `devicename` = part number complet | `devicename: "GW2A-18C"` | **FIXÉ** (validé réf nand2mario) |
| 3 | `EX3794` doublons | `DramSoCTop.v` auto-suffisant (38 mods) | 2 fichiers seulement | **FIXÉ** (idem `UartSoC.v`, 55 mods) |
| 4 | Sweep 31 LUT silencieux | Optimizer Gowin replie du vivant (Migen) | `syn_keep` sur 2357 nets (SUG550 §5.8) | **CONTOURNÉ** (1915 reg/3588 LUT) |
| 5 | `CK0021` OSER4_MEM→OBUF (DQS) | `ELVDS_IOBUF` Migen illégal EDA | tristate ref-style + N par tuile SSTL15D | **FIXÉ** |
| 6 | Cascade NL0002 (39 mods SoC) | Core mort → `init_done=0` → reset tenu | Fix 4+5 (cause racine) | **RÉSOLU**, 0 NL0002, 8360 LUT |
| 7 | `PR2017` resetN sur T10 | Pin SSPI dédiée | M10 (via T2/A6, cf. §3) | **FIXÉ** |
| 8 | `CT1136` bank 4 vccio | LVCMOS33 en bank DDR 1.5V | Reset en bank 2 (M10) | **FIXÉ** |
| 9 | Silence UART DRAM sur silicium | **OUVERT** : `init_done=0` ou `pll_locked=0` ? | P1/P3 (§8) | **EN COURS** |

## 5. Comparaison MNIST : OSS vs propriétaire (même RTL)

| | OSS (Yosys→nextpnr→gowin_pack, `--no-dsp`) | Gowin factory (`gw_sh`) |
|---|---|---|
| LUT | 10088 LUT4 | 5719 LUT + 1599 ALU (= 7318) |
| Regs | 5197 FF | 4599 |
| BSRAM | 27 | 31 |
| DSP | **25 MULT18X18 aussi** (`synth.json`) | 25 MULT18X18 + 5 MULT9X9 |
| Fmax/contrainte | 33,67 MHz (> 27 ✓) | pas de `.sdc` (TA1132 bénin) |
| Silicium | **PASS** (ping, upload, HW==NumPy) | **PASS** (même selftest) |

- Correction d'une erreur d'analyse : `--no-dsp` est **partiel**
  (bloque les chaînes MAC/ALU, pas les MULT nus — 25 MULT18X18 des deux
  côtés, prouvé dans `synth.json`). L'écart LUT restant (10088 vs 7318)
  = comptage pré-pack vs post-synthèse + split LUT/ALU + mappeur/packer
  vendeur plus dense. **Normal, pas une anomalie.**
- Conclusion : la factory propriétaire est prouvée sur RTL connu-bon,
  silicium inclus. Le différentiel Yosys/Gowin est un bruit de fond,
  pas un signal.

## 6. DRAM côté vendeur (chiffres à retenir)

- Synth : 8360 LUT / 4765 reg / 43 BSRAM (94%) / 8 DSP, 0 NL0002, 0 ERROR.
  Détail : `soc_acc` 4246 LUT (matmul 2261…), `soc_dram/core` 3513 LUT.
- PnR : **PASS**, IOLOGIC 16 IDES4_MEM + 20 OSER4_MEM + 25 OSER4 +
  25 IODELAY, **DLL 1/4, DQS 2/9, rPLL 1/4** (le gap nextpnr `_MEM`
  est contourné). `.fs` 7093 KiB. CLS 64%.
- Réf Yosys : core seul ~3k LUT ; SoC complet `--synth-only` 8873 LUT.
- rPLL généré : `FCLKIN 27.0`, `DEVICE GW2A-18C`, `ODIV_SEL 16`,
  `CLKIN=clkin_signal` ; CLKDIV `DIV_MODE 2`, `RESETN=~reset0`.
- Bémols : BSRAM 94% (3 libres — GAO affamé), pas de `.sdc`
  (la preuve = l'UART), ZQCL 15µs vs 19µs requis (marginal, à revoir).

## 7. Validation silicium (A/B)

- `examples/Mnist/inference.py --selftest-only` (terminal, pas de Gradio) :
  ping `V`→`0x01`, upload 2936B, inférence, comparé NumPy → **PASS COM8**.
- `examples/UniversalScale/silicon_validate.py [--ping-only]` : **FAIL**,
  aucun port ne répond (carte flashée `DRAM.fs`, `Finished` Programmer).
- Interprétation : carte + USB-UART + H11 + protocole + outils = prouvés
  par le contrôle. Le silence est interne au design DRAM.
- Note bitstream : le `DramSoCTop.v` embarqué tourne **UniversalScaleDemo**
  (64 sorties), pas MNIST (`DramSoCGen` dit Mnistw4a8 mais le Verilog
  synthétisé est antérieur — **recompiler avant release**).

## 8. Diagnostic du silence (état P1)

- Différentiel UartSoC (répond) vs DramSoCTop (muet), même bridge/pins/clk :
  câblage bridge **identique** ; seuls deltas = `DdrAdapter` vs
  `BramAdapter`, pads DDR, et **reset += `!initSync1`**.
- Waits d'init audités : 50000/10000/200 ×2 cycles @27MHz ≈ 3,7ms/740µs/
  15µs — compteurs inconditionnels, FSM sans état bloquant (revu) :
  `init_done` **doit** s'activer en quelques ms **si** `sys_clk` vit et
  `sys_rst` se relâche.
- Donc : `sys_rst` tenu (`pll_locked=0` ?) ou `sys_clk` morte (rPLL/CLKDIV),
  ou `resetN` (M10) à 0 sur PCB malgré pull-up. M10 = bille BGA,
  non sondable au multimètre.
- Écartés (avec raison) : EX3780 (style Migen normal), horloge morte en
  synthèse (clocks vivantes au rapport), cercle reset↔init (pas de notion
  de blocage en synthèse), wrapper hiérarchique (namespaces par fragment
  divergents), keeps ciblés (un keep ne rend pas un constant dynamique).
- GAO : `.gao` créé, 4 signaux trouvés (`soc_dram/core_pll_locked`,
  `soc_dram/core_init_done`, `reset`, `io_uartTx`), trigger M1
  (`init_done==1`), re-PnR (`ao_0.fs`) OK — mais **lecture pend**
  (trigger 1 comme 0, pas de Force Trigger trouvé). Même signature que
  les déboires USB → cause hôte probable, pas design. À re-tenter sur
  USB sain avant d'incriminer les cœurs GAO.
- Apicula : écarté pour ce debug (fuses ≠ état vivant) ; reste pertinent
  pour le reverse nextpnr (`docs/plan-reverse.md`).

## 9. Plan

- **P1 — Audit Verilog-truth** (sans GUI/carte) : connectivité
  `rPLL→sys2x→CLKDIV→sys`, chaînes `locked/POR→sys_rst`, sampling
  `initDoneRaw→initSync1→reset`, séquence CKE/reset_n vs JEDEC,
  relecture patch DQS, corriger ZQCL (200→512+ cycles).
- **P2 — FAIT** (factory + presets `dram`/`mnist`, verdicts PASS).
- **P3 — Readout sans JTAG** : debug UART (statut/blink `pll_locked`/
  `init_done` dans le RTL) + extension `silicon_validate.py` ;
  re-tenter GAO sur USB sain en parallèle (zéro rebuild).
- **Release** : recompile Spinal (dqs_n déjà retiré des sources +
  choisir le modèle : MNIST ou UniversalScaleDemo), regen factory,
  `.sdc` 27 MHz, flash, validation bit-exacte, archiver le `.fs` qui
  marche + `gowin_unpack` pour nextpnr.
- **Dette** : minimiser le keep-set (timing/area), BSRAM 94% à surveiller,
  `DRAM.fs` racine du repo (copie de travail, trier avant commit).

## 10. Reproduction (rappel §2)

```powershell
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset dram
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset mnist --workdir E:\eda-factory\MNIST --skip-compile
.\.venv\Scripts\python.exe examples\UniversalScale\silicon_validate.py --ping-only
python .\examples\Mnist\inference.py --selftest-only
```
Artefacts : `E:/eda-factory/*`, `out/mnist-control-rtl/`,
`hw_build/mnist-control/top.fs`, `out/rtl-dram-backup/`,
`out/eda-ui-logs/`, `out/gsruns/` (runs headless GowinSynthesis).
