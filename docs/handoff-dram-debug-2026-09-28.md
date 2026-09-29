# HANDOFF — Debug lecture DRAM (sim Verilator + silicium) — 28/09/2026 soir

> Pour l'agent qui reprend : tout le contexte nécessaire est ici.
> Docs précédents : `docs/dram-alive-2026-09-27.md` (victoire silicium),
> `docs/audit-silicium-2026-09-27.md`, `docs/eda-bringup-session.md`.

## 0. TL;DR

- **Silicium VIVANT** : ping `V` OK, uploads OK, trigger OK, `accBusy=1`
  (`S=0x83`, AR pending). L'accélérateur démarre et attend des lectures
  DRAM qui ne reviennent jamais. Bitstream : `E:/eda-factory/DRAM/impl/pnr/DRAM.fs`
  (**SRAM volatile !** tout rebranchement efface → reflasher, 4 s).
- **Sim Verilator FONCTIONNELLE** : `litedram_core.v` + `prim_sim.v` Gowin +
  modèle Micron x16 tournent sous SpinalSim/Mill (étage A PASS).
- **Bug isolé en sim** : après `init_done=1`, une écriture AXI single-beat
  est acceptée (aw/w ready) mais `b_valid` ne vient jamais. En cause :
  **le crossbar ne transfère rien vers les bankmachines**
  (`req_valid[7:0]=0` sur tout le run) alors que le contrôleur est vivant
  (storm PREA+REF visible au modèle). Suspect n°1 = stall crossbar
  (entrée `new_port_cmd_valid` → sortie req), pas le PHY read-capture.
- Fichier de travail : `spinalML/test/src/spinalML/memory/litedram/LiteDramCoreSimTest.scala`
  (étages A+B, pré-processing Verilator, moniteur `bind`, TB AXI).

## 1. Reproduction (Windows powershell, depuis E:/spinalML)

```powershell
# 1. Regen RTL (après modif Spinal ou dram/gen/gowin_gen.py) :
.\.venv\Scripts\python.exe cli\main.py compile tests\universal\UniversalScaleDemo.scala --dram
# 2. Sim (étages A+B, ~3 min, SANS retry inutile) :
.\.venv\Scripts\python.exe cli\main.py test-all -k LiteDramCoreSim --retry 0
# Log : out\test_reports\spinalML.memory.litedram.LiteDramCoreSimTest.log
# 3. Silicium (reflash SRAM + validation) :
.\.venv\Scripts\python.exe cli\main.py flash E:\eda-factory\DRAM\impl\pnr\DRAM.fs
.\.venv\Scripts\python.exe examples\UniversalScale\silicon_validate.py [--ping-only]
.\.venv\Scripts\python.exe examples\UniversalScale\read_heartbeat.py 8 [COM8]
# 4. Rebuild factory après modif RTL :
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset dram --skip-compile
```

## 2. Étage A : PASS (à garder vert)

- TB `SimTop` : core seul, pads DDR en l'air, AXI idle, `prim_sim.v` Gowin.
- Verdict : `pll_locked=1` puis `init_done=1` (~250 kcycles). Prouve en sim :
  fix R1 (domaines release `cd_sys_rel`, fini l'auto-maintien) + CRG E2
  (rPLL main style réf, sys sur CLKOUTD, sans CLKDIV) + séquenceur JEDEC.
- Adaptations Verilator (générées au runtime dans `dram/out/sim/`,
  **gitignored**, rien à committer) :
  - `prim_sim_norpll.v` : `prim_sim.v` IDE moins `module rPLL` (oscillateur
    à `#delay` = boucle comb une fois ignorés) + `GSR.GSRO` → `1'b1`
    (pas d'instance GSR : refs non résolues sous Verilator).
  - `rpll_stub.v` (maison) : horloges idéales + LOCK après 256 coups.
  - Flags : `--no-timing`, `-Wno-COMBDLY/IEEEMAYDEPRECATE/REALCVT/IMPLICIT/
    PINMISSING/CASEINCOMPLETE/MISINDENT/LATCH/MULTIDRIVEN/SELRANGE/CASEX`
    (tous revus bénins : modèles vendeur bruyants).
- **PIÈGE CONNU du stub rPLL : il tourne à DEMI-RÉGIME** (toggle par front
  = ÷2 : sys@13.5MHz, sys2x@27MHz au lieu de 27/54). Ratios 1:2 préservés
  donc fonctionnellement OK, mais temps absolus ×2 (ex: init_done à
  ~11.5ms sim au lieu de ~7ms ; preuvres : POR 65536 coups = 2.43ms ✓,
  MR2 du modèle à 11.40ms ✓). À corriger en passant le TB à 54MHz
  (`forkStimulus(18519)`) si les temps absolus comptent un jour
  (checks timing modèle, actuellement OFF).

## 3. Étage B : état du bug (LE front)

- TB `SimTopB` : core + modèle Micron x16 (`ddr3_wrap.v` généré :
  `dqs_n = ~dqs_p`, `defparam` strict_timing=0, strict_mrbits=1,
  `STOP_ON_ERROR=0` = mode OBSERVE). Écritures single-beat ×4 adresses
  (mots 0x0, 0x1, 0x100, 0x10000), puis lectures comparées.
- **Verdict actuel** : `write b_valid timeout @word 0`. Init OK (~11.5ms
  sim demi-régime), aw/w acceptés, puis rien.
- **Preuves (moniteur `bind lite_dbg`, trigger sur changement, exhaustif)** :
  - `req_valid[7:0] = 0` sur TOUT le run (v4, 2661 échantillons) :
    aucun bankmachine ne reçoit rien.
  - Contrôleur vivant : storm `Precharge All` + `Refresh` périodique
    (~156µs sim) vu au modèle ; `core_cmd_valid` collé à 1 (unité refresh).
  - `bm0 state` alterne 0/4 (activité refresh), jamais de traitement.
  - Run v3 (autre layout sonde) : `new_port_cmd_valid` blippe + DFI avec
    strobes WR+PRE par période, TOUJOURS sans `b`, sans Activate modèle.
  - **Divergence v3/v4 non expliquée** : v4 ne voit jamais `npcv=1`.
    Piste : trafic fantôme à t=0 (le modèle loggue WRITE/READ spurious
    banques 1,6,7 au démarrage) qui coince des `litedramcore_lockedN`
    différemment à chaque run → non-déterminisme. À vérifier.
- **Le modèle NE PEUT PAS valider notre init** (limite dure) :
  `litedram/init.py:1639` : `if (dll_en) init_step++` — avec MR1 DLL-off
  (notre cas : `dll_off=True`, CL=CWL=6, ODT off), le tracker d'init reste
  bloqué → tout Activate/Refresh = "Initialization sequence is not
  complete". Donc : séquenceur prouvé par l'absence d'erreurs MR + MRS
  bien reçues (MR2 loggué avec PASR/ASR/SRT), mais PAS de validation
  init complète possible avec ce modèle en DLL-off.
- Bruits identifiés (ne pas chasser) : erreurs t=0 (X de démarrage),
  `DQS bit 31`/`DQ bit 159` (boucles check 64 entrées fixes du modèle),
  `CKE must be inactive when RST_N` à 2.43ms (transitoire POR, ordre
  d'événements zéro-délai), `RST_N pulse width` (compteur X),
  DQS-idle "latching edge" (pas de trafic), backlog FTDI (~1440 frames
  après 12min = normal, pas une horloge rapide).

## 4. Suspects classés (pour le prochain run)

1. **Stall crossbar** (`new_port_cmd_valid` → `req_valid` = 0) : sonder
   l'entrée esclave du crossbar + `litedramcore_lockedN` + `roundrobin*_grant`
   (noms exacts à grep dans `dram/out/litedram_core.v`), et la timeline du
   premier blip `npcv` (11.5ms).
2. **Row-tracking bankmachine** : WR sans ACT préalable vu en v3 (hit
   fantôme après reset ? le storm PRE ne met pas à jour le tracking ?).
   Sonder `litedramcore_bankmachineN_state` du banc servi + `req_addr`.
3. **Nondéterminisme t=0** : fantômes banques 1,6,7 → `lockedN` coincés ?
   Sonder `lockedN` + comparer deux runs.
4. **Config AXI2Native** (framing `first/last` constants, `size=3`,
   `len=0`) : relire `litedram/frontend/axi.py` (venv managé) si 1-3
   innocentent le crossbar.
5. **Variante B1 DLL-on** (plus tard) : `SIM_DLL_ON` dans `gowin_gen.py` +
   `dram-gen --out dram/out/sim --name litedram_core_dllon` + 2e BlackBox.
   Validerait le data-path digital complet (le modèle supporte DLL-on).
   Ne prouverait PAS le timing de capture silicium (domaine différent).

## 5. Fichiers

- À committer : `spinalML/test/src/spinalML/memory/litedram/LiteDramCoreSimTest.scala`,
  `spinalML/src/spinalML/memory/litedram/LiteDramCore.scala` (`corePath`
  absolu pour Mill), `examples/UniversalScale/{read_heartbeat.py,
  silicon_validate.py}` (poll S), `dram/gen/gowin_gen.py` (CRG E2, dbg),
  `spinalML/src/spinalML/io/DramSoCTop.scala` (heartbeat), docs.
- Gitignored (générés) : `dram/out/sim/*`, `dram/out/litedram_core.v`.
- Refs hors repo (E:/refs, ne pas committer) : `ddr3-tang-primer-20k`
  (nand2mario), `ddr3-buttercutter/ddr3.v` (modèle v1.74 classique),
  `ddr3-micron-model/1024Mb_ddr3_parameters.vh`, `ddr3-tejchavan` (inutile,
  SV-only — supprimable). Override possible : `DDR3_MODEL_DIR`,
  `GOWIN_SIMLIB_DIR`, `SPINALML_ROOT`.
- Modèle : `+define+den1024Mb+sg25E`, x16 par défaut, `check_strict_*`
  et `STOP_ON_ERROR` forcés par `defparam` dans le wrapper généré.

## 6. Prochaines actions concrètes

1. LITEDBG v5 : ajouter `req_valid` est déjà là — ajouter côté entrée
   (`new_port_cmd_*` payload + `roundrobin*_grant` + `lockedN[7:0]`) et
   capturer la fenêtre 11.4-11.6ms (premier blip).
2. Selon résultat : fixer (crossbar ? row-track ? framing ?) → rerun
   jusqu'à `write→read→compare` PASS sur les 4 mots.
3. Puis : retour silicium (reflash + `silicon_validate.py` complet).
4. Doc : porter le root-cause final dans `docs/dram-alive-2026-09-27.md`.
