# Issue nextpnr — cellules DDR Gowin manquantes (brouillon bilingue)

> **À quoi sert ce fichier** : la section anglaise ci-dessous (§2) est le texte
> **prêt à copier-coller** dans une issue GitHub `YosysHQ/nextpnr` (tracker
> public, cf. `docs/nextpnr-ddr-gap.md` §8 pour le contexte de recherche).
> La section française (§1) explique ce qu'on y raconte et pourquoi c'est
> formulé comme ça. Dossier technique complet : `docs/nextpnr-ddr-gap.md`.

## 1. Explication (français — pour toi)

**Où poster** : https://github.com/YosysHQ/nextpnr/issues → « New issue ».
Choisis un titre proche de celui proposé (il contient les mots-clés que les
mainteneurs scannent : `himbaechel/gowin`, noms des cellules, DDR).

**Ce que le texte anglais contient, dans l'ordre** :
1. Le setup exact (outil, versions, chip, design) pour que ce soit
   reproductible sans poser de question.
2. L'erreur exacte + à quelle étape (placement, après synthèse OK).
3. La preuve que ce n'est pas notre design : pinout prouvé en flow vendeur
   (cible litex-boards), synthèse propre, IOLOGIC simples OK.
4. La cause racine avec **refs fichiers/lignes** (`gowin.h`, `pack_iologic.cc`,
   `gowin.cc`) — c'est ce qui transforme un « ça marche pas » en issue
   actionnable : le mainteneur voit immédiatement le trou.
5. La liste exhaustive des 4 types manquants (pas juste celui de l'erreur),
   pour éviter 3 allers-retours.
6. Ce qu'on a déjà écarté (techmap plain = invalide électriquement,
   DHCEN = notre bug corrigé, IO_TYPE = sans effet) — pour ne pas recevoir
   ces suggestions en réponse.
7. L'offre concrète : repro minimal + dispo pour tester un patch sur notre
   hardware (Tang Primer 20K physique). C'est ce qui motive un mainteneur.

**Ce qu'on ne met PAS** : pas de pavé SpinalHDL/LiteDRAM (bruit pour eux),
pas de demande de délai, pas de critique. Reste factuel et testable.

**Après postage** : colle l'URL de l'issue dans `docs/liteDRAM.md` §6 et
ici même (section 3 ci-dessous).

## 2. Issue text (English — copy-paste ready)

**Title:** `himbaechel/gowin: DDR memory-interface cells OSER4_MEM/IDES4_MEM/DQS/DLL have no packing support (GW2A-18, LiteDRAM design)`

**Body:**

```markdown
### Environment
- nextpnr-himbaechel `0.11.1-19-g8dbcee5c` (oss-cad-suite 2026-09-06)
- Yosys `0.68+195`, `synth_gowin`
- Device `GW2A-LV18PG256C8/I7` (`--vopt family=GW2A-18`), Sipeed Tang Primer 20K
- Design: LiteDRAM 2026.08 standalone core (GW2DDRPHY, DDR3 x16, 1:2 ratio,
  27 MHz sys) + SoC logic, ~13.9k LUT4 / 8.8k DFF post-synth (fits: 66%/56%)

### What works
- `synth_gowin` completes cleanly; all hardened DDR cells survive synthesis
  (`rPLL`, `CLKDIV`, `DHCEN`-free, `DQS`x2, `OSER4`x25, `OSER4_MEM`x20,
  `IDES4_MEM`x16, `IODELAY`x25, `ELVDS_IOBUF`x2, `DLL`x1).
- Plain `ODDR`/`OSER4`/`IDDR`/`IDES4` outputs pack and constrain fine.
- The pinout is proven with the vendor flow (LiteX-Boards Tang Primer 20K
  target, DDR3 x16 footprint), so this is not a pin/bank mistake.

### Error
Place stage fails on the first memory-interface serializer:
```

ERROR: Unable to place cell 'soc_dram.core.OSER4_MEM_9', no BELs remaining to implement cell type 'OSER4_MEM'
```

### Root cause (read from current main sources)
- `himbaechel/uarch/gowin/gowin.h`: `type_is_iologico` covers
  `{ODDR, ODDRC, OSER4, OSER8, OSER10, OVIDEO, EMPTY}` and
  `type_is_iologici` covers `{IDDR, IDDRC, IDES4, IDES8, IDES10, IVIDEO,
  EMPTY}` — the `_MEM` variants are in neither list.
- `himbaechel/uarch/gowin/pack_iologic.cc` (`pack_iologic()`): the three
  dispatch branches handle exactly those lists, so `OSER4_MEM`/`IDES4_MEM`
  cells are never packed to `IOLOGICA/B` BELs and reach placement abstract,
  where the empty `OSER4_MEM` bucket fails.
- `grep id_DQS / id_DLL himbaechel/uarch/gowin/*.cc *.h` returns nothing:
  `DQS` (x2 in this design) and `DLL` (x1) have no handling either and
  would fail right after.
- `gowin_arch_gen.py` only creates `IOLOGICA/B` (+`IDES16`/`OSER16`) BELs,
  so there is nowhere for these cell types to go today.

### Full missing-cell inventory for a DDR link
| Cell | Count here | Needed for |
|---|---|---|
| `OSER4_MEM` | 20 | DQ/DM/DQS write serialization (DQS-aligned) |
| `IDES4_MEM` | 16 | DQ read capture (DQS-aligned) |
| `DQS` | 2 | DQSBUFM helpers, one per byte group |
| `DLL` | 1 | DDRDLLA delay calibration |

### Already ruled out
- Not a constraints issue: per-bit `IO_LOC` + `IO_TYPE=SSTL15/SSTL15D`
  verified applied (zero `Cell ... not found`), same failure.
- Not a clocking issue: an earlier `DHCEN 25/24` overflow was our own CRG
  gate; without it the report shows `DHCEN 0/24` and placement still fails
  on `OSER4_MEM`.
- Mapping `_MEM` -> plain `OSER4`/`IDES4` in techmap is electrically invalid
  for DDR: it drops the DQS alignment (`TCLK`@270, `CALIB`), violating
  tDS/tDH (writes) and read capture. Not a viable workaround.

### Repro & offer
I can provide a minimal failing case (single `OSER4_MEM` on a GW2A-18 pin
fails the same way) and I own the physical board above, so I can test any
patch end-to-end (PnR + `gowin_pack` + on-silicon DRAM traffic) and report
back. Happy to bisect, test, or help validate fuse-level details against
vendor-EDA reference bitstreams (`gowin_unpack`) if that helps scope the
`IOLOGIC` MEM-mode work.
```

## 3. Suivi

- [ ] Issue postée le : ____  — URL : ____
- [ ] Réponse upstream : ____
