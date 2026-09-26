# Plan : validation silicium via Gowin EDA (Windows/Linux + WSL)

> **Objectif** : obtenir un bitstream `DramSoCTop` qui tourne sur la Tang
> Primer 20K **sans attendre** le fix nextpnr, en faisant synth+PnR avec les
> outils vendeur. Nos Verilog sont compatibles vendeur par construction
> (mêmes cellules que LiteX envoie à Gowin EDA) et on n'a **aucun IP vendeur**
> à régénérer (LiteDRAM remplace le DDR IP) : cas EDA le plus simple.
> **Contexte** : `docs/liteDRAM.md`, gap open-source : `docs/nextpnr-ddr-gap.md`.
> **Situation de départ (26/09)** : pas de compte Gowin, install à faire côté
> Linux/WSL.

## 0. Prérequis compte + licence (à faire en premier, délai inconnu)

1. Créer un compte sur https://www.gowinsemi.com (Support → Download).
2. Télécharger **Gowin EDA Linux** (version récente, ex. YunYuan/V1.9.x).
   Poids ~1-2Go, prévoir ~5Go libres (install + projet + builds).
3. Demander la **licence FPGA Design** (gratuite pour GW2A en général) via
   « Apply License ». Sans licence, la synthèse/PnR EDA ne tourne pas.
4. Pendant l'attente : préparer les étapes 1-2 ci-dessous (tout est faisable
   sans licence).

## 1. Installation côté WSL (points de vigilance)

- Installer sous WSL2 (ex. `~/gowin/`) : l'EDA Linux tourne nativement.
- **Espace disque** : vérifier `df -h` (≥5Go libres).
- **USB vers WSL** : pour flasher depuis WSL, pont USB obligatoire —
  `usbipd-win` côté Windows + `usbipd attach` du debugger FTDI/JTAG, puis
  `openFPGALoader` dans WSL voit la carte. **Alternative sans friction** :
  générer le `.fs` sous WSL, flasher depuis Windows natif avec
  `openFPGALoader -b tangprimer20k` (le CLI `spinalml flash` sait déjà le
  faire côté Windows). Recommandé pour commencer.
- Interop fichiers : travailler dans `~` WSL (pas `/mnt/c`, perfs), copier
  le `.fs` final vers Windows si flash natif.

## 2. Projet EDA (Verilog-only, pas d'IP)

1. Nouveau projet FPGA : device **GW2A-LV18PG256C8/I7**, top `DramSoCTop`.
2. Sources à ajouter (générées par notre flow, cf. `docs/liteDRAM.md` §9) :
   - `DramSoCTop.v` (+ `Model.v`, chaîne UART : `UartRx/Tx/Bridge`) —
     via `compile … --dram` (ou le `rtl/` du repo après un build `--dram`).
   - `dram/out/litedram_core.v` — via `dram-gen`.
   - **NE PAS ajouter** `dram/prims/gowin_bb.v` (shim Yosys uniquement ;
     EDA connaît nativement DLL/IODELAY/ELVDS_IOBUF — un stub les masquerait).
3. Contraintes physiques : convertir `hw_build/<board>/pins.cst`
   (généré par `build --dram`, 52 IO_LOC per-bit déjà validés) au format
   attendu par EDA (`.cst` Gowin : même syntaxe `IO_LOC`/`IO_PORT`
   pour l'essentiel — vérifier `IO_TYPE=SSTL15/SSTL15D` acceptés, sinon
   équivalent FloorPlanner).
4. Contraintes timing (`.sdc`/`.fdc`) : horloge 27MHz sur `clk` (+ `clk_p`
   DDR générée en interne, contrainte auto via la PLL/DLL normalement).
5. Lancer synth + PnR EDA, relever : LUT/FF/BSRAM/DSP, Fmax, warnings DQS/DLL.

## 3. Bitstream + flash + validation fonctionnelle

1. Générer le `.fs` (SRAM, chargement ~1s).
2. Flasher : `openFPGALoader -b tangprimer20k <top>.fs` (natif Windows ou WSL+usbipd).
3. Validation hôte (protocole `docs/uart_bridge.md`, outillage `uart_host.py`) :
   - charger image + poids (dépassant la BRAM pour forcer le chemin DRAM),
   - inférence, comparer aux logits `ModelReplica` (bit-exact attendu),
   - noter bande passante/latence apparentes si mesurables.
4. Critère de succès : inférence bit-exact sur un modèle qui **ne tient pas
   en BRAM** → le lien DRAM fonctionne (init JEDEC gateware + PHY + contrôleur).

## 4. Livrables réutilisables (alimentent le reverse, cf. `docs/plan-reverse.md`)

- Le bitstream `.fs` qui marche + le projet EDA archivé (version EDA notée).
- `gowin_unpack -d GW2A-18 -o ref.v <top>.fs` (apicula) : lecture de comment
  l'EDA a configuré IOLOGIC/DQS/DLL pour nos pins exacts → carburant du
  support nextpnr (bien mieux que du fuzzing aveugle).
- Mettre à jour `boards/tang-primer-20k.json` (`memory.ddr.present=true`,
  `size_bytes=134217728`, `controller="litedram-gw2ddrphy"`) et
  `docs/roadmap_board.md` une fois le silicium vert.

## Risques connus

- Licence : délai/validation côté Gowin, hors de notre contrôle.
- WSL+USB : `usbipd` parfois capricieux → fallback flash Windows natif.
- EDA peut râler sur des détails (noms de pins diff, `IO_TYPE` exacts,
  horloge DDR) : itérer sur les warnings, ils sont normalement explicites.
