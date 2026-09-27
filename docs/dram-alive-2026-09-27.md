# DRAM vivante sur silicium — 27/09/2026 (soir)

> Le FPGA parle : ping `V` OK, uploads OK, trigger OK, `accBusy=1`.
> Reste : l'inférence ne rend rien (0/64B) — l'accélérateur attend des
> lectures DRAM qui ne reviennent jamais. Journal détaillé :
> `docs/eda-bringup-session.md`, audit précédent :
> `docs/audit-silicium-2026-09-27.md` (le §7 sur le modèle est CORRIGÉ §4
> ci-dessous).

## 1. Victoire du jour

- `silicon_validate.py --ping-only` : `ping V: OK on COM8`, `status S: 0x01`.
- Validation complète : uploads 1024B + 65600B sans erreur, CSR programmés,
  trigger pris en compte → `poll S: 0x83` stable (`accBusy=1`,
  `AR pending`). Le lien, le protocole, les CSR, le trigger : **prouvés**.
- Preuve que le reset est libéré et `init_done=1` : le heartbeat s'est tu
  tout seul (il ne parle que quand `init_done=0`).

## 2. Root causes (silence → vie)

| # | Cause | Preuve silicium | Fix |
|---|---|---|---|
| R1 | Release synchrone **auto-maintenu** : `_rst0/_rst1` (init 1) vivaient dans le domaine dont le reset (`cd_sys.rst`) qu'ils pilotent → `if (sys_rst) regs <= 1` éternel. Idem `_rst2x_*`. Le commentaire "équivalent en pratique" était faux. | Heartbeat `sys_rst=1, step=0` avec `locked=1` | Domaines release `cd_sys_rel`/`cd_sys2x_rel` (reset POR `~por_done`, indépendant des sorties) — `dram/gen/gowin_gen.py` CRG |
| R2 | Les **deux horloges PLL mortes** malgré `LOCK=1` : après R1, `req`/`s2x_rst` toujours à 1, `tick` gelé. Lowering LiteX `GW2APLL` (`FCLKIN "27.0"`, `FDLY 15`, `PSDA "0000"`) + `CLKDIV` maison. | Inchangé après R1 | rPLL main **style nand2mario** (`FCLKIN "27"`, `FDLY 0`, `PSDA "0100"`, VCO 864MHz in-spec, CLKOUT 54MHz) + `sys` sur `CLKOUTD ÷2` (`DYN_SDIV_SEL 2`), **CLKDIV supprimé** |
| R3 | `crg.reset` **non-drivé** (alimentait `CLKDIV.RESETN` + chaînes reset) | Lecture revue | Tié explicitement à 0 + commentaire anti-optimizer |

- Méthode : heartbeat UART boot-domain (`0xA5 + status + rstinfo`, `examples/UniversalScale/read_heartbeat.py`), statut : bit0 lock, bit1 init, bit2 sys_rst, bits[7:3] step idx ; rstinfo : bit7 sys-tick, bit6 por, bit5 req, bit4 creset, bit3 s2x_rst, bit0 reset_n. Chaque hypothèse a été tranchée par UN bit.
- Pièges écartés : M10 innocenté (`reset_n=1` lu), PLL params OK (VCO 864), `--no-dsp` partiel (normal), dissection `.fs` inutile pour du vivant, `DRAM.vg` chiffré (illisible, ne pas auditer), compteur "frame #1425" = backlog FTDI drainé, pas horloge rapide.

## 3. Front actuel : lecture DRAM pendue

- `poll S: 0x83` stable : `accBusy=1`, `accDone=0`, `outValid=0`, **AR pending** (bit7). L'accélérateur a démarré et attend une lecture.
- Écritures : non vérifiées (pas de readback dans le protocole — `W` ne rend rien). Le hang est côté READ (PHY capture DQS/IDES sans training ? ZQCL 15µs vs 19µs ? ODT ?).
- Référence qui marche : nand2mario fait write-leveling + read-calibration + ODT dynamique ; notre PHY n'en fait aucun (init JEDEC gateware seule).

## 4. Corrections au dossier

- **Modèle embarqué** : `compile tests/universal/UniversalScaleDemo.scala --dram` → AutoRunner enveloppe `{comp_name}` (l'accélérateur du fichier) dans `DramSoCTop` (`cli/spinalml_cli/compile_flow.py:184`). `DramSoCGen.scala` (Mnistw4a8) est une entrée SÉPARÉE, non utilisée ici. Le bitstream contient bien UniversalScale (64 sorties). La note contraire de l'audit §7 est fausse.
- **Comparaison `examples/Mnist/inference.py`** : même protocole (`V/S/C/W/R`), mêmes CSR (`0x00` trigger, `0x08` img, `0x0C` poids) ; MNIST ajoute `0x30` (échelle dequant runtime, inutile en I8 pur). Ordre poids : MNIST = payload pré-packé du `.npz` ; Universal = slice-transpose script (`p*Ks*N+o*Ks+kl`), **prouvé bit-exact vs sim** (30350 cyc, `dev=0.0`). Le layout n'est pas suspect ; le non-démarrage non plus (`busy=1`). Reste la lecture physique.

## 5. Prochaines pistes (ordonnées)

1. **Simuler `litedram_core`** (jamais simulé, seulement synthétisé) contre modèle DDR LiteDRAM : attrape un hang FSM sans round-trip silicium.
2. **Read-training / capture** : délais IODELAY, polarité DQS, `BURSTDET`, ODT ; comparer registres PHY avec nand2mario à 54MHz (eux tournent à 400MHz avec calibration — à 54MHz on devrait passer sans, mais la capture reste le suspect n°1).
3. **ZQCL** 200→512+ cycles (marginal, pas cher).
4. Re-tenter GAO sur USB sain (signaux : `init_done`, `rddata_valid`, FSM controller) ; garder le heartbeat en option debug (coût 300 LUT).

## 6. Commandes (Windows)

```powershell
# SRAM volatile ! tout rebranchement efface -> reflasher (4s)
.\.venv\Scripts\python.exe cli\main.py flash E:\eda-factory\DRAM\impl\pnr\DRAM.fs
.\.venv\Scripts\python.exe examples\UniversalScale\read_heartbeat.py 8 [COM8]
.\.venv\Scripts\python.exe examples\UniversalScale\silicon_validate.py [--ping-only]
# Regen complète après modif Spinal/Migen :
.\.venv\Scripts\python.exe cli\main.py compile tests\universal\UniversalScaleDemo.scala --dram
.\.venv\Scripts\python.exe scripts\eda_flow.py --preset dram --skip-compile
```

## 7. Fichiers touchés aujourd'hui

- `spinalML/src/spinalML/io/DramSoCTop.scala` : heartbeat (`hb`, mux TX, `dbgStepRaw`/`dbgRstRaw`).
- `spinalML/src/spinalML/memory/litedram/{LiteDramCore,LiteDramAxiBridge}.scala` : ports `dbg_step`/`dbg_rst`.
- `dram/gen/gowin_gen.py` : `DramInitSequencer.step_idx`, `dbg_step`/`dbg_rst`, tie `crg.reset=0`, domaines release, **CRG rPLL main** (plus de `GW2APLL`/`CLKDIV`).
- `examples/UniversalScale/{read_heartbeat.py` (nouveau), `silicon_validate.py` (poll S après trigger).
- `E:/refs/ddr3-tang-primer-20k` (clone ref, hors repo).
