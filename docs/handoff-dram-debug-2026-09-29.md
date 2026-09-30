# HANDOFF — Debug DRAM sim (stage B/D) — 29/09/2026

> Reprend ici sans relire tout l'historique. Contexte antérieur :
> `docs/handoff-dram-debug-2026-09-28.md` (silicium vivant, infra sim),
> `docs/dram-alive-2026-09-27.md`, `docs/eda-bringup-session.md`.
> Ce fichier = état exact + preuves + prochaines actions.

## 0. Où on en est (TL;DR)

- **Plan écriture : PROUVÉ de bout en bout.** Nos octets arrivent sur les
  pads DQ/DQS dans l'ordre (CDEF..FEDC), donc AXI → converter → crossbar →
  bankmachine → contrôleur → PHY : tout le chemin write est sain.
- **Plan lecture : la donnée revient mais décalée/répétée.** Dernier verdict :
  `mismatch @word 0: got 456789AB456789AB exp 0123456789ABCDEF`.
  La capture PHY ne prend que 2 beats en commençant au beat 1 (`[1,2,1,2]`),
  puis répète. Suspect : fenêtre de capture lecture du PHY
  (DQS gate / RADDR / IDELAY en DLL-off), PAS le contrôleur.
- **Estimation : ~85 % — 1 à 3 runs de la fin.** Reste : cartographier la
  fenêtre (lire word1), comparer les taps lecture vs réf nand2mario,
  ajuster → vert. Puis stage B (Micron+force, même TB) → silicium.

## 1. Commandes (powershell, depuis E:/spinalML)

```powershell
# Tests (1 seul workspace Verilator par suite ; ~85-130 s/run) :
.\.venv\Scripts\python.exe cli\main.py test-all -k LiteDramCoreDfi --retry 0  # stage D (répondeur, rapide)
.\.venv\Scripts\python.exe cli\main.py test-all -k LiteDramCoreAxi --retry 0  # stage B (Micron, JEDEC fidèle)
.\.venv\Scripts\python.exe cli\main.py test-all -k LiteDramCoreInit --retry 0 # stage A (init, ~1 min, vert)
# Mill direct équivalent :
C:\Users\leona\.spinalml_tools\mill.bat --no-server --disable-ticker spinalML.test.testOnly spinalML.memory.litedram.LiteDramCoreDfiTest
# Logs : out/test_reports/spinalML.memory.litedram.LiteDramCore{Dfi,Axi,Init}Test.log
# Regen core (défaut = silicium POR=65536) :
.\.venv\Scripts\python.exe cli\main.py dram-gen [--por-cycles 256]
```

## 2. Fichiers (ce qui compte)

- `spinalML/test/src/spinalML/memory/litedram/LiteDramCoreSimTest.scala`
  = `GowinSimPrep` (stub rPLL plein régime 54/27, prim_sim_norpll,
  wrapper Micron + `force init_done`, répondeur `dfi_resp`, sonde v12)
  + `LiteDramCoreInitTest` (stage A, vert, ne plus toucher).
- `.../LiteDramCoreAxiTest.scala` = stage B seul (SimTopB + Micron).
- `.../LiteDramCoreDfiTest.scala` = stage D seul (SimTopD + dfi_resp,
  burst 2-beat + lectures single, fire-and-forget partout).
- `dram/gen/gowin_gen.py` : `--por-cycles` (SIM-ONLY, warning imprimé ;
  chemins silicium ne le passent jamais). `dram/out/litedram_core.v`
  actuel = POR 65536 (silicium-safe, régénéré sans flag le 29/09).
- `cli/spinalml_cli/dram_cmd.py` (`por_cycles` param), `cli.py`
  (option `--por-cycles` sur `dram-gen` uniquement).
- Refs : `E:/refs/ddr3-tang-primer-20k` (nand2mario, lectures DLL-off OK
  sur même carte), `E:/refs/ddr3-buttercutter/ddr3.v`,
  `E:/refs/ddr3-micron-model/*.vh`. Venv libs :
  `C:/Users/leona/.spinalml_tools/.venv/Lib/site-packages/litedram/...`
  (`frontend/axi.py`, `frontend/adapter.py`, `core/bankmachine.py`,
  `core/crossbar.py`, `phy/gw2ddrphy.py`).

## 3. Corrections déjà faites (ne pas refaire)

1. **Stub rPLL plein régime** (`rpllStub`) : TB à 54 MHz
   (`forkStimulus(18519)`), sys2x=54 toggle both-edges, sys=27 posedge
   (÷2). Raison : toggler both-edges donne F/1 pas 2F (une version a
   roulé sys2x à 27 MHz, tout le timing DFI/DQS faux — mesuré 37 ns).
2. **TB fire-and-forget** : pulses 2 cycles TB (= 1 période sys,
   exactly-once) puis drop, JAMAIS de hold (un hold crée un duplicata
   fantôme qui wedge burst2beat : `aw_r=0` pour toujours).
3. **Timeouts bornés partout** (30k cycles b/r, 2000 reopen) + asserts
   localisés (`w/aw/b/ar/r timeout`, `aw never reopened`). Un
   `waitSamplingWhere` nu = pendaison infinie ; un poll 200k = ~15 min
   (vécu). Ne jamais régresser là-dessus.
4. **Bursts SoC-like** (pas de single-beats) : single 64-bit + last=1 =
   demi-mot incomplet que LiteX committe zero-paddé (CLOBBER entre
   paires, documenté dans l'UpConverter). Le TB écrit en bursts INCR
   2-beat (last au dernier beat uniquement), `wValid` tenu SANS trous
   (les trous cassent le merge du convertisseur).
5. **Répondeur `dfi_resp`** : CAM 16 entrées tag→mot 128b ; WR capturé
   sur fronts DQS (offset calibré = 2 : préambule 1 CK, prouvé par
   raw `[0,0,CDEF..FEDC,0,0]`) ; **silence sur MISS** (un drive sur
   MISS contentionne un WR voisin) ; **pre-drive beat0 + index+1**
   (IDES échantillonne pré-front, NBA).
6. **Micron (stage B)** : `force mem.init_done=1` dans `ddr3_wrap.v`
   (sinon le modèle jette tout en DLL-off), `STOP_ON_ERROR=0`,
   `MAX_MEM` désactivé, `+define+den1024Mb+sg25E`.
7. **Sonde v12** : 51 bits échantillonnés CHAQUE front sys2x dans une
   fenêtre bornée (init_done + 15000), pas de trigger sur changement
   (les pulses 1-cycle sont invisibles sinon). Layout dans le commentaire
   du code. Formats de temps inconsistants (`t=.. ps` vs `t=..`) :
   parser avec `LITEDBG t=([\d.]+)\s*(ps)?\s*([01]{N})`.
8. **Split fichiers** : 1 suite = 1 classe = 1 fichier (Init/Axi/Dfi),
   filtre `-k` matche le FQCN. `GowinSimPrep` partagé.

## 4. Preuves clés (dans les logs)

- Mot 128-bit OUR sur pads, 8 beats en ordre :
  `WR tag=0 beats=12 ... raw=0000_0000_fedc_ba98_7654_3210_0123_4567_89ab_cdef_0000_0000`
  (offset 2 = préambule ; le mot stocké = `{FEDC..CDEF}` = D2|D1 exact).
- Lecture : `RD tag=0 hit=1` + même donnée rejouée ; TB reçoit
  `456789AB456789AB` = beats rword[1],[2] répétés (avant pre-drive :
  `[1],[0]` répétés — le pre-drive a bien shifté de +1).
- Convertisseur innocenté (cycle NEW→FILL→COMMIT→CMD→NEW, `wdatav=1`
  avec nos moitiés `cdef`/`babe`), ACT sur pads (stage B/Micron),
  `b`/`r` fires, reopen OK.
- Stage B (Micron+force) : ACT + WR + RD exécutés (erreurs restantes =
  artefacts DLL-off du modèle : tXPR/tDLLK/tRTW en OBSERVE).

## 5. Prochaines actions (ordre)

1. **Cartographier la fenêtre** : lire word1 (high 64) AVANT word0
   (ou ne pas asserter au premier mismatch, printer les deux) —
   1 ligne + 1 run. Attendu : `[5,6,5,6]`-like si fenêtre 2-beats.
2. **Comparer les taps lecture** (`gw2ddrphy.py` : IDELAY, DQS gate,
   RADDR/retards, `read_latency`) avec la réf nand2mario (lecture
   seule, 0 run). Candidats : gate qui ouvre 1 beat tard + se referme
   après 2 beats, ou RADDR qui boucle sur 2 slots.
3. **Ajuster** (config PHY ou workaround TB/simu) → TB vert stage D.
4. **Stage B** (même TB, Micron+force) → preuve JEDEC-fidèle.
5. **Silicium** : reflash + `silicon_validate.py` (le S=0x83 pourrait
   être la même capture lecture côté carte !).
6. Reporter le root-cause final dans `docs/dram-alive-2026-09-27.md`.

## 6. Pièges connus (ne pas retomber dedans)

- `waitSamplingWhere` nu / caps >30k / pulse 1 cycle TB à 54 MHz
  (il faut 2 cycles = 1 période sys) / trous dans `wValid` /
  `last=1` par beat / `awValid` tenu (duplicata fantôme) /
  `bResp`/`rData` échantillonnés hors pulse (stale).
- `%0t` sans unité stable ; compter les bits du vecteur sonde
  (troncation silencieuse si mismatch) ; `stageDbgMon`/`prepare`/
  wrappers réécrits à chaque run (pas de cache stale).
- SimWorkspace locké après kill brutal (`AccessDeniedException` sur le
  `.dll`) → killer les java orphelins + supprimer le sandbox.
- `dram/out/litedram_core.v` POR=256 = SIM-ONLY (régénérer sans flag
  avant tout build silicium ; `compile --dram` le fait déjà).
- Bruit modèle t=0 / DQS-idle / backlog FTDI : bénin, documenté.
