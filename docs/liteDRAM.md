# LiteDRAM sur spinalML — journal de bord

> **Statut** : session en cours (ouverte le 26/09/2026).
> **Objectif** : brancher l'infrastructure spinalML sur LiteDRAM, focus **Gowin
> Tang Primer 20K** (20K LUT, 128MiB DDR3) pour valider la chaîne complète.
> **Règles actées** : structure modulaire multi-FPGA (`dram/` + `memory/litedram/`),
> ASIC hors-scope v1, commentaires de code en anglais, discussion en français,
> `.venv` racine intouché, `dram.txt` seule source de vérité des pins Python.
>
> Ce document est le **journal vivant** de la session : chaque étape, chaque
> souci rencontré et chaque patch appliqué y sont consignés (et continueront
> de l'être).

## 0. Périmètre acté (questions du 26/09)

| Décision | Choix |
|---|---|
| Chip DDR | SK Hynix **H5TC1G63EFR** (1Gb, 64M×16, 128MiB) — révision board de Léo (pas le IMD d'autres révisions, même footprint x16 donc mêmes pins) |
| Connexion RTL | **BlackBox + Top wrapper** (`LiteDramCore` + `DramSoCTop`), pas de concaténation brute, pas de SoC LiteX |
| ASIC | Hors-scope, juste un seam (`require(!isAsic)` explicite) |
| Pins Python | `litex@2026.08` + `litedram@2026.08` explicites (tags vérifiés existants) ; `migen` figé sur commit (plus de release PyPI, dev sur `git.m-labs.hk`) |
| Dossier gen | Nouveau top-level `dram/` (pas `scripts/`, pas `spinalml_cli/` — ce dernier tourne dans le `.venv` racine **sans** litex) |
| Flags CLI (P3) | `--dram` (pas `--with-dram`), sur `compile` **et** `build` (qui accepte dir / `.v` / `.scala`) |
| Doc | Ce fichier, tenu à jour à chaque étape |

## 1. P0 — Environnement (`requirements/dram.txt`)

- `dram.txt` avant : `migen==0.9.2 / litex==2024.12 / litedram==2024.12` (PyPI, 18 mois de retard).
- Après : 3 pins git figés (reproductibilité bit-exact) :
  ```
  migen @ git+https://git.m-labs.hk/M-Labs/migen.git@beffe831bf1a691eaebf9ddbb4660f8b3e7fd964
  litex @ git+https://github.com/enjoy-digital/litex.git@2026.08
  litedram @ git+https://github.com/enjoy-digital/litedram.git@2026.08
  ```
- SHA migen résolu via `git ls-remote` (`beffe831bf` = HEAD/master au 26/09/2026).
- **Leçon apprise** : `uv pip install -r` résout bien les dépendances transitives
  (`litedram/setup.py` tire `litex` sans borne), mais sans pin explicite = drift
  silencieux entre deux `setup`. De plus `pyenv.collect_pins()` ne parse que
  `==/>=/...` → les lignes `@ git+...` sont **invisibles pour `spinalml pylibs`**
  (dette : étendre la regex en phase d'implémentation).
- `migen` vendored par oss-cad-suite ? **Non vérifiable** (`~/.spinalml_tools`
  était supprimé) ; de toute façon sans impact (nos flows n'importent jamais le
  Python de la suite : `PYTHONHOME` sanitizé, `runners/env.py`). À vérifier au
  prochain setup : `ls ~/.spinalml_tools/oss-cad-suite/lib/python*/site-packages/ | grep -i migen`.
- `spinalml setup` relancé de zéro (validé OK : CLI + managed env verts).
  Le `.venv` racine n'est jamais touché par les flows DRAM (règle single-runtime).
- Environnement de dev : après hésitation `/tmp` vs managed env, tout le travail
  se fait depuis `~/.spinalml_tools/.venv/bin/python` (le scratch `/tmp/litedram-dev`
  n'a servi qu'à inspecter `litex-boards`).

## 2. P1 — Générateur Gowin (`dram/`)

Fichiers créés (commentaires en anglais) :

| Fichier | Rôle |
|---|---|
| `dram/gen/gowin_gen.py` | Générateur : CRG (GW2APLL 27→54MHz) + `GW2DDRPHY` + `LiteDRAMCore` (contrôleur+crossbar) + port AXI 64b + séquenceur d'init JEDEC → `dram/out/litedram_core.v` |
| `dram/gen/sdram_modules.py` | `H5TC1G63EFR(DDR3Module)` : géométrie 8×8192×1024 (= `MT41K64M16`), timings 1Gb conservateurs (le DDR tourne à 54MHz, tout quantize à 1-2 cycles) |
| `dram/gen/validate.py` | Checks config : clés requises, module connu, `sys_clk` sane, taille géométrie == `size_bytes`, contrat AXI 64b |
| `dram/configs/tang-primer-20k.yml` | `sys_clk 27MHz`, `GW2A-18/GW2A-LV18PG256C8/I7`, Hynix, 134217728 bytes |
| `dram/prims/gowin_bb.v` | **Source versionnée** (pas générée !) : stubs `(* blackbox *)` DLL/IODELAY/ELVDS_IOBUF — voir §3 |

Choix structurants :
- **`litedram.gen` upstream inutilisable pour Gowin** (que SDR/ECP5/Xilinx) →
  composition manuelle des blocs LiteDRAM (pattern `add_sdram` de LiteX).
- **`sys_clk = 27MHz`** ( = horloge board) : pas de CDC sur le lien AXI, DDR à
  54MHz, `dll_off=True` automatique (2×sys ≤ 125MHz, même règle que la cible
  litex-boards). Monter plus tard = pont CDC côté SpinalHDL (suivi acté).
- **Séquenceur d'init JEDEC en gateware** (`DramInitSequencer`) : sans CPU/BIOS,
  personne ne joue la séquence DDR3 → FSM sur `DFIInjector.ext_dfi`
  (mode hardware-control, `_control.sel` reset à 1), valeurs MR reprises de
  `litedram.init.get_ddr3_phy_init_sequence`, waits `max(2×délai, 4)` sys cycles
  (marge ×2 + plancher tMRD), `init_done` bascule le mux DFI vers le contrôleur.
  **Sans ça : silence radio garanti sur silicium.**
- Port AXI via `crossbar.get_port(data_width=64)` (adaptation de largeur
  intégrée au crossbar).
- Pinout DDR repris de `litex_boards.platforms.sipeed_tang_primer_20k`
  (footprint x16 commun aux révisions IMD/Hynix) ; `clk27=H11`, UART T13/M11
  identiques au `.cst` spinalML existant.

### Bugs croisés pendant P1 (tous corrigés)
1. `Signal(max(max(...)))` au lieu de `Signal(max=...)` : `max(201)` → `TypeError:
   'int' object is not iterable`. Une heure de fausse piste « shadowing » pour
   une coquille (le `if False` et le `finalize_dispatch` morts ont été nettoyés
   au passage).
2. Interface AXI (`Record`, pas `Module`) enregistrée comme submodule →
   `AttributeError: get_fragment_called`. Fix : attribut simple.
3. `AsyncResetSynchronizer` : `lower()` lève **toujours** sans plateforme LiteX →
   remplacé par une chaîne de 2 flops plain-Migen (sources quasi-statiques :
   POR + lock PLL, équivalent en pratique).
4. Records AXI dans `ios` de `verilog.convert` (tri par `duid`) → aplatisseur
   via `Record.flatten()`.
5. **`self.pll = ...` ne collecte PAS le submodule** dans ce Migen (pas de
   `__setattr__` magique) → `self.submodules.pll`, sinon rPLL absent du Verilog
   **sans erreur**. Piège majeur : le design semblait complet avec `sys2x`
   flottant et `locked=0`.
6. `create_clkout(..., with_reset=True)` ajoute aussi un `AsyncResetSynchronizer`
   → `with_reset=False` + reset explicite du domaine `sys2x_i` (auto-amorcé sur
   init bitstream).

## 3. Souci flow open-source : cellules Gowin manquantes (RÉSOLU en synthèse)

- `synth_gowin` échouait : `DLL`, `IODELAY` (×25), `ELVDS_IOBUF` (×2) absents de
  `cells_sim.v` (vérifié : présents = rPLL/DHCEN/CLKDIV/DQS/OSER4_MEM/IDES4_MEM/IOBUF).
- **Preuve par sonde** (`/tmp`, hors repo) : design minimal + stubs blackbox →
  Yosys OK **et nextpnr-himbaechel accepte les cellules** (échec uniquement sur
  contraintes manquantes, attendu). Le flow open-source place donc ces cellules.
- `dram/prims/gowin_bb.v` déclare les 3 cellules en `(* blackbox *)` avec les
  signatures exactes émises par LiteDRAM 2026.08. **Aucun compromis fonctionnel** :
  sur silicium ce sont de vraies cellules (la DLL calibre pour de vrai via
  `GW2DDRPHYInit`, DQS reste différentiel). Le stub est un shim de synthèse.
- Couplage documenté dans l'en-tête du fichier : à re-vérifier après tout bump
  LiteDRAM (`grep -A9 "^DLL \|^IODELAY \|^ELVDS_IOBUF " dram/out/litedram_core.v`).
- `.gitignore` : `*.v` ignorait aussi le stub → exception `!dram/prims/gowin_bb.v`
  ajoutée (vérifié par `git add --dry-run` : 5 sources suivies, `dram/out/*.v` ignoré).
- Synthèse complète `litedram_core` seul : **7s, ~3k LUT (15%), 1880 FF, 0 BSRAM,
  0 DSP**, toutes les cellules durcies préservées (DLL×1, DQS×2, OSER/IDES×36, rPLL…).

## 4. P2 — Wrapper Scala (FAIT, tests verts)

| Fichier | Rôle |
|---|---|
| `memory/litedram/LiteDramCore.scala` | BlackBox aux ~100 ports exacts (`noIoPrefix`, `setDefinitionName("litedram_core")`, `addRTLPath("dram/out/litedram_core.v")`) |
| `memory/litedram/LiteDramAxiBridge.scala` | AXI4 standard → streaming LiteX, **possède** le core ; `aw/ar_first/last=1`, compteur de beats + **bypass même-cycle** (`w_first/w_last/w_param_id` utilisent les champs AW frais si `aw.fire` coïncide), `addr[29:3]`, `len` resizé 4→8b |
| `io/DramSoCTop.scala` | Miroir `UartSoC` (protocole hôte/CSR inchangés) : acc + `DdrAdapter` + pont + UART ; reset tenu jusqu'à `init_done` synchronisé ; pads `ddram_*` exposés ; `require(!isAsic)` |
| `io/DramSoCGen.scala` | Entrypoint `runMain` (même modèle Mnistw4a8 que `UartSoCGen` → comparabilité) |
| `test/.../memory/litedram/LiteDramBlackBoxTest.scala` | Élaboration + asserts `litedram_core`/`DdrAdapter`/`ddram_a` |

### Pièges SpinalHDL rencontrés (référence future)
1. **Horloge → BlackBox** : `x.io.clk := clockDomain.clock` (ou `cd.clock`)
   pendouille **toujours** (les fils d'horloge sont hors netlist signaux).
   Seule voie : `core.mapClockDomain(clock = core.io.clk27)` (vérifié dans les
   sources 1.15.0 locales `~/SpinalHDL-1.15.0/.../BlackBox.scala:157`).
2. **`RegNext(False)` + `:=`** = double driver (OVERLAP) → `Reg(Bool()) init(False)` nu.
3. **Registres du reset-gate en domaine racine/BOOT** : dans `cd`, le reset ne se
   relâche jamais (deadlock). Ils vivent dans un `ClockingArea(BOOT)` (pas de
   port reset émis, liste de ports minimale vérifiée : clk/resetN/uart/ddram_*).
4. **Domaine BOOT inexistant en ASIC** → `require(!target.isAsic)` assumé
   (LiteDRAM = FPGA-only, acté §0).
5. Champs SpinalHDL 1.15.0 : `addr(29 downto 3)` rend déjà `UInt` (pas de
   `.asUInt`) ; `aw.payload.lock` est `Bits` → `lock(0)` ; inout BlackBox =
   `inout(Analog(Bits(...)))`.
6. `compile` CLI fonctionne déjà sur `DramSoCGen.scala` (bonus P3 : `DramSoCTop.v`
   généré avec ports `io_*` propres).

## 5. Mesures ressources (question LUT de Léo) — RÉPONDU le 26/09

Comparaison iso-modèle (MNIST w4a8, niveau synth primitives Yosys) :

| Design | LUT-prim | ALU | FF | BSRAM | MULT18 |
|---|---|---|---|---|---|
| `UartSoC` (BRAM) | 7524 | 2572 | 5197 | 0 | 25 |
| `DramSoCTop` (DRAM) | 9158 | 2786 | 6934 | 0 | 25 |
| **Taxe DRAM** | **+1634 (+22%)** | +214 | +1737 | — | — |

Verdict : la crainte « 1/4 du budget pour la DRAM » est **levée** — la taxe
mesurée vaut ~+8% du budget LUT (1,6k/20k). Le 13,9k LUT4 packés (66%) du plein
PnR vient du SoC+modèle lui-même (~11-12k, déjà là en BRAM), pas du contrôleur.
Le modèle MNIST w4a8 reste le poste dominant (DSP 52%).

## 6. Problème PnR (26/09) : DHCEN RÉSOLU, OSER4_MEM OUVERT

- **DHCEN 25/24 → 0/24 : RÉSOLU.** La cause était **notre** DHCEN de CRG
  (`sys2x` gaté par `stop` non drivé) : en sa présence, le packer HCLK de
  nextpnr construisait tout l'arbre d'horloges DDR à travers des DHCEN
  (un par OSER4). Fix : `sys2x = sys2x_i` direct dans `GowinDramCRG`
  (le `stop`/freeze DDRDLLA n'a pas besoin d'une gate fabric à 54MHz).
  Fichier : `dram/gen/gowin_gen.py` (commentaire NB sur place).
- **IO_TYPE sans effet sur le fond** : les 39 `Cell io_ddram_* not found`
  venaient d'`IO_PORT` sans index (le parseur nextpnr veut du per-bit comme
  `IO_LOC`). Corrigé en sonde (`IO_PORT "io_ddram_dq[0]" IO_TYPE=SSTL15;`) :
  0 « not found », mais chiffres PnR **identiques**. Piste écartée comme cause
  racine (reste à faire proprement en P3 dans le `.cst` board).
- **RESTE : `OSER4_MEM_9` — `no BELs remaining`.** État des lieux :
  - `pack_iologic()` de nextpnr-himbaechel traite `ODDR/OSER4/OSER8/IDDR/IDES4…`
    mais **pas les variantes `_MEM`** (vérifié dans les sources upstream
    `pack_iologic.cc`) : les 20 `OSER4_MEM` + 16 `IDES4_MEM` de LiteDRAM
    restent abstraits et le placeur échoue au 10ème (`_9`).
  - Le pinout est pourtant prouvé en flow vendeur (cible litex-boards,
    doc Gowin TN662 : chaque bank GW2A a des ressources DQ) → **gap de
    modélisation apicula/nextpnr probable, pas bug de notre design**.
  - Pistes restantes : (a) consommation de BELs IOLOGIC par les dummies
    `create_aux_iologic_cell` ; (b) remonter upstream (issue nextpnr) ;
    (c) contournement via `cli/spinalml_cli/patches/` si une règle de
    placement simple émerge (jamais de fork LiteDRAM).
- Fichiers de debug (hors repo) : `/tmp/dram_soc_synth.json`,
  `/tmp/ddrprobe/pnr_full*.log`, `/tmp/ddrprobe/ddr_pins.cst`.

## 7. P3 FAIT (26/09) : hooks CLI `--dram`

- `spinalml dram-gen --board tang-primer-20k` (nouveau, `cli/spinalml_cli/dram_cmd.py`,
  wrapper fin sans import litex → subprocess managed python) : validate + gen OK.
- `compile --dram` (`compile_flow.py`) : implique `--soc` pour les Accelerator,
  AutoRunner émet `DramSoCTop` au lieu de `UartSoC` ; `dram-gen` tourne après
  l'élaboration. Vérifié : `compile examples/Mnist/Model.scala --dram` →
  `DramSoCTop.v` + `Model.v`. Legacy sans `--dram` inchangé (`UartSoC.v`).
- `build --dram` (`build_runner.py`) : forward `dram` à `compile` pour les
  sources `.scala`, `dram-gen` + ajout `dram/out/litedram_core.v` et
  `dram/prims/gowin_bb.v` à la liste Yosys, top défaut `DramSoCTop`.
- E2E : `build examples/Mnist/Model.scala --dram --synth-only` → Yosys 33s,
  **LUT 7661 (37%), FF 6942 (45%), BSRAM 0, DSP 25 (52%)** (modèle MNIST CNN
  complet, cf. §5 pour la taxe iso-modèle).
- `.cst` : pins DDR ajoutés à `boards/constraints/tang-primer-20k.cst`
  (signaux `ddram_*`, SSTL15/SSTL15D) ; `adapt_constraints_for_ports` étendu
  (expansion des bus per-bit, résolu par nom de base + alias — legacy
  clk/uart identique). `pins.cst` adapté vérifié : 52 IO_LOC, bons IO_TYPE.
- Reste (silicium/P4) : le PnR bute toujours sur `OSER4_MEM` (cf. §6) ;
  le flow `--dram` est complet et valide jusqu'à la synthèse.

## 8. Journal des patches

| # | Fichier | Cible | Raison | Condition de retrait |
|---|---|---|---|---|
| D1 | `dram/prims/gowin_bb.v` | Yosys `synth_gowin` | `DLL`/`IODELAY`/`ELVDS_IOBUF` absents de `cells_sim.v` (pas de modèle, pas d'erreur de principe côté PnR — prouvé §3) | Yosys modélise ces cellules Gowin (vérifier à chaque bump oss-cad-suite) |
| D2 | `filter_unique_verilog_files` (`build_runner.py`) | build `--dram` | la regex `^\s*module` ratait `` (* blackbox *) module DLL`` → `gowin_bb.v` droppé comme « redondant » → `hierarchy -check` KO | upstream Yosys (pas lié aux versions) : garder tant que des stubs à attributs existent |
| — | `cli/spinalml_cli/patches/` (existant) | apicula/nextpnr | Mécanisme prévu si §6 exige un contournement (ex. mapping DHCEN/OSER) | Chaque patch porte sa condition (convention du dossier) |

## 9. Commandes utiles (session)

```bash
.venv/bin/python cli/main.py setup                    # recrée outils + managed env (ne touche PAS .venv)
.venv/bin/python cli/main.py test-all -k "LiteDramBlackBox"
~/.spinalml_tools/.venv/bin/python dram/gen/validate.py dram/configs/tang-primer-20k.yml
~/.spinalml_tools/.venv/bin/python dram/gen/gowin_gen.py --config dram/configs/tang-primer-20k.yml --out dram/out
.venv/bin/python cli/main.py compile spinalML/src/spinalML/io/DramSoCGen.scala -o /tmp/x --no-chain
```
