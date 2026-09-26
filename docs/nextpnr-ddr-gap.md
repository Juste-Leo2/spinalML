# Gap DDR Gowin dans nextpnr-himbaechel — dossier technique

> **Objet** : tout ce qu'il faut pour comprendre pourquoi le bitstream
> `DramSoCTop` + LiteDRAM ne passe pas le placement sous chaîne open-source,
> et ce qui manque exactement côté nextpnr.
> **Contexte projet** : voir `docs/liteDRAM.md` (journal de la session).
> **Conclusion en une phrase** : le design est correct et la synthèse passe,
> mais nextpnr-himbaechel ne sait placer ni `OSER4_MEM`/`IDES4_MEM`, ni `DQS`,
> ni `DLL` — quatre types de cellules indispensables au chemin DDR durci.

## 1. Versions et périmètre

| Pièce | Version exacte |
|---|---|
| Yosys | 0.68+195 (oss-cad-suite du 2026-09-06) |
| nextpnr-himbaechel | 0.11.1-19-g8dbcee5c (même suite) |
| LiteDRAM | 2026.08 (`c454a44`) |
| LiteX | 2026.8 (`b6ae9e0`, tag vérifié) |
| Migen | commit `beffe831bf` (pas de release PyPI en amont) |
| Cible | GW2A-LV18PG256C8/I7 (`--vopt family=GW2A-18`), Sipeed Tang Primer 20K |
| DRAM | H5TC1G63EFR 1Gb (64M×16), DDR3 à 54MHz (sys 27MHz, ratio 1:2, DLL-off) |

## 2. Où la chaîne casse (tableau des trois erreurs)

| # | Étape | Message exact | Statut |
|---|---|---|---|
| E1 | Yosys `synth_gowin` (`hierarchy -check`) | `ERROR: Module '\DLL' referenced in module '\litedram_core' in cell '\DLL' is not part of the design.` | **Résolue** (patch D1) |
| E2 | nextpnr, rapport d'utilisation | `DHCEN: 25/24 104%` | **Résolue** (c'était notre gate) |
| E3 | nextpnr, placement | `ERROR: Unable to place cell 'soc_dram.core.OSER4_MEM_9', no BELs remaining to implement cell type 'OSER4_MEM'` | **Ouverte = ce dossier** |

Après E3, `gowin_pack` n'est jamais atteint. Tout l'amont (compile Scala,
`dram-gen`, synthèse 33s, sim) est vert.

## 3. E1 — `DLL` inconnu de Yosys (résolue, rappel)

La bibliothèque Gowin de Yosys (`cells_sim.v`) modélise `rPLL`, `DHCEN`,
`CLKDIV`, `DQS`, `OSER4_MEM`, `IDES4_MEM`, `IOBUF`… mais ni `DLL` (DDRDLLA),
ni `IODELAY`, ni `ELVDS_IOBUF`, tous trois instanciés par `GW2DDRPHY`.
Comme `hierarchy -check` rejette tout module non défini, la synthèse
échouait avant même de commencer.

**Résolution (patch D1, `dram/prims/gowin_bb.v`)** : les trois cellules sont
déclarées `(* blackbox *)` avec les signatures exactes émises par LiteDRAM
2026.08. Yosys les traverse, nextpnr les place (preuve par sonde minimale :
le PnR accepte les trois types, voir §6). Sur silicium ce sont de vraies
cellules (la DLL calibre pour de vrai via `GW2DDRPHYInit`, DQS reste
différentiel) : aucun compromis fonctionnel, juste un shim de synthèse.

## 4. E2 — `DHCEN: 25/24` (résolue, rappel)

Le premier PnR complet demandait 25 DHCEN pour 24 BELs — exactement le nombre
d'`OSER4` du design (25 sorties DDR) **plus** le DHCEN de notre CRG
(`sys2x` gaté par un `stop` non drivé, copié de la cible litex-boards).
En présence de cette gate, le packer HCLK de nextpnr construisait tout
l'arbre d'horloges DDR à travers des DHCEN.

**Résolution** : `sys2x = sys2x_i` direct dans `GowinDramCRG`
(`dram/gen/gowin_gen.py`, commentaire NB sur place) — le freeze DDRDLLA n'a
pas besoin d'une gate fabric à 54MHz. Résultat : **DHCEN 0/24**. Le design
n'a intrinsèquement besoin d'aucun DHCEN.

## 5. E3 — `OSER4_MEM` implaçable (le vrai gap)

### 5.1. Log complet du point de rupture

```
Info: Pack IODELAY...
Info: Pack IO logic...
Info:                 LUT4:   13887/  20736    66%
Info:                  DFF:    8836/  15552    56%
Info:                BSRAM:      11/     46    23%
Info:            MULT18X18:      25/     48    52%
Info:                 rPLL:       1/      4    25%
Info:                DHCEN:       0/     24     0%
...
Info: Running custom HCLK placer...
Info: Placed 0 cells based on constraints.
ERROR: Unable to place cell 'soc_dram.core.OSER4_MEM_9', no BELs remaining to implement cell type 'OSER4_MEM'
```

### 5.2. Inventaire des cellules DDR du design (post-synthèse)

| Cellule | Qté | Rôle | Connue de Yosys | Plaçable nextpnr |
|---|---|---|---|---|
| `OSER4` | 25 | Sorties addr/cmd (serialiseurs 4:1) | oui | oui (packées, aucune erreur) |
| `IODELAY` | 25 | Délais statiques sorties | via D1 | oui (`pack_iodelay` ✓) |
| `OSER4_MEM` | 20 | Écriture DQ (16) + DM (2) + DQS (2), clockés DQS | oui | **NON** |
| `IDES4_MEM` | 16 | Lecture DQ (capture calée DQS) | oui | **NON** |
| `DQS` | 2 | Helpers DQSBUFM par groupe d'octets | oui | **NON** |
| `DLL` | 1 | DDRDLLA, calibration des délais | via D1 | **NON** |
| `ELVDS_IOBUF` | 2 | DQS différentiel vers les pins | via D1 | oui (sonde §6) |
| `rPLL`/`CLKDIV`/`IOBUF` | ok | Horloges + IO simples | oui | oui |

### 5.3. Cause racine (lue dans les sources nextpnr courantes)

- `himbaechel/uarch/gowin/gowin.h:51-61` :
  `type_is_iologico` = {ODDR, ODDRC, OSER4, OSER8, OSER10, OVIDEO, EMPTY} et
  `type_is_iologici` = {IDDR, IDDRC, IDES4, IDES8, IDES10, IVIDEO, EMPTY}.
  **Les variantes `_MEM` n'y figurent pas.**
- `himbaechel/uarch/gowin/pack_iologic.cc:638-648` (`pack_iologic()`) : les
  trois branches traitent exactement ces listes — **`OSER4_MEM`/`IDES4_MEM`
  tombent dans aucun cas** et restent des cellules abstraites.
- `himbaechel/uarch/gowin/gowin.cc:1071` (`getBelBucketForCellType`) : sans
  mapping, ces cellules gardent un bucket à leur nom, sans BELs en face
  (`gowin_arch_gen.py` ne crée que des BELs `IOLOGICA/B`, `IDES16`/`OSER16`).
- `grep -rn "id_DQS\|id_DLL" himbaechel/uarch/gowin/*.cc *.h` : **zéro
  occurrence** — `DQS` et `DLL` n'ont aucun traitement non plus (ils
  échoueraient juste après les `_MEM`).
- Mécanisme de l'erreur : le placeur ne trouve aucun BEL candidat pour le
  bucket `OSER4_MEM` → `no BELs remaining` (au 10ème par ordre interne ;
  les 9 premiers ne sont pas vraiment placés non plus).

### 5.4. Pourquoi ce n'est pas un bug de notre design

- Le **pinout est prouvé en flow vendeur** : repris tel quel de la cible
  `litex-boards` Tang Primer 20K (qui build avec `--toolchain gowin`), avec
  les groupes d'octets DQ/DQS contigus par construction.
- La doc Gowin TN662 (référence DDR2/3 sur GW2A-LV18PG256) confirme que chaque
  bank GW2A porte des ressources DQ : le silicium a ces ressources, c'est le
  **modèle apicula/nextpnr qui ne les expose pas** pour le chemin `_MEM`.
- Le support « I/O DDR and SERDES » annoncé par nextpnr couvre les briques
  simples (ODDR/OSER/IDDR/IDES, utilisées par ex. en vidéo/MIPI), pas le
  chemin mémoire complet (`_MEM` + `DQS` + `DLL`) qu'exige tout PHY DDR
  type LiteDRAM. **Tout** design LiteDRAM-GW2A est concerné, pas le nôtre
  en particulier.

### 5.5. Contournement « techmap `_MEM` → plain » étudié et rejeté

Convertir `OSER4_MEM`→`OSER4` / `IDES4_MEM`→`IDES4` en techmap Yosys perd
l'alignement DQS (entrée `TCLK` à 270°, `CALIB`) : DQS et DQ sortiraient sur
les mêmes fronts FCLK (skew ≈ 0) alors que DDR3 exige DQS **centré à 90°**
(`tDS`/`tDH` en écriture, `tDQSS`, capture en lecture). Le recréer en fabric
exigerait de modifier le PHY (fork LiteDRAM — interdit par nos règles).
**Pas de patch repo possible sans casser le lien DDR.** Le mécanisme
`cli/spinalml_cli/patches/` reste disponible si une règle de placement
simple émerge côté upstream.

## 6. Preuves et reproductions

- **Sonde cellules** (prouve que Yosys+nextpnr acceptent DLL/IODELAY/
  ELVDS_IOBUF en blackbox) : design de 15 lignes + `cells_bb.v`, Yosys OK,
  nextpnr OK jusqu'au packing IOB (échec attendu : contraintes manquantes).
- **Repro complète** : `compile examples/Mnist/Model.scala --dram` puis
  `build … --dram` (s'arrête à `--synth-only` en gate P3) ; en sonde,
  `yosys … synth_gowin` + `nextpnr-himbaechel --device GW2A-LV18PG256C8/I7
  --vopt family=GW2A-18 --vopt cst=<pins DDR>` → E3 ci-dessus.
- Fichiers de debug (hors repo, session du 26/09) : `/tmp/dram_soc_synth.json`,
  `/tmp/ddrprobe/pnr_full*.log`, `/tmp/ddrprobe/ddr_pins.cst` (52 IO_LOC,
  IO_TYPE SSTL15/SSTL15D per-bit vérifiés).

## 7. Le commit `c4fbb55` (« singleton vector [0] », 26/09) — pourquoi ça n'aide pas

- **Ce qu'il fait** (20 lignes, `cst.cc` uniquement) : un port vecteur d'1 bit
  contraint comme `NAME[0]` retrouve la cellule nommée `NAME` dans le JSON
  (via `getCellForPinConstraint()`, comme ice40/ecp5 déjà), et la branche
  `_n` d'une paire diff dérive de la cellule `_p` résolue. Issu de #1791/#1804.
  Absent de notre snapshot (oss-cad-suite du 06/09).
- **Pourquoi sans effet ici** : (a) notre « Cell not found » venait d'`IO_PORT`
  sans index sur des bus **multi-bits** — le fix ne couvre que les singletons
  `[0]`, déjà contourné par nos `IO_PORT` per-bit ; (b) notre bloqueur est au
  **placement** (pack/place), pas au parsing CST ; (c) nos paires diff sont
  des bus 2 bits, pas des singletons.
- **Bonne nouvelle indirecte** : le coin CST Gowin bouge le jour même via une
  issue utilisateur → l'upstream est réactif sur ce périmètre, de bon augure
  pour une remontée du gap `_MEM`.

## 8. Impact et voies silicium

- Bloqué : bitstream open-source du design DRAM (PnR après synthèse OK en 33s).
- Non bloqué : sim bit-exact, synthèse + rapports, fit-check, preuve à grande
  échelle en sim (piste suivante de la session).
- Voie silicium alternative : **Gowin EDA** (gratuit, Linux) fait synth+PnR de
  nos Verilog (`DramSoCTop.v` + `litedram_core.v`, sans les stubs) — le design
  est compatible vendeur par construction (mêmes cellules que LiteX y envoie).
