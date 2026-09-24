# Audit — Refactoring Python & préparation LiteDRAM

> **Statut** : plan acté le 22/09/2026, à exécuter avant l'intégration LiteDRAM.
> **Périmètre** : gestion de l'environnement Python, nettoyage pré-LiteDRAM,
> et sortie des patchs toolchain codés en dur (`build_runner.py`).
> Priorité n°1 définie avec le maintainer : la santé du code d'abord,
> LiteDRAM ensuite. Rien n'est encore implémenté — ce document est la spec.

## 1. État actuel (inventaire au 22/09/2026)

### 1.1 Environnement Python
- `.venv` racine, Python 3.12.3, **sans module pip** (créé artisanalement,
  probablement via `uv venv`), mélangeant CLI (`typer`, `requests`, `rich`
  via `cli/requirements.txt`) et tests HW (`pytest 8`, `cocotb 1.9.2`,
  `numpy`, `cocotb-test`, `cocotbext-axi` via `/requirements.txt`).
- `cocotb` exclu Windows par markers (`sys_platform != 'win32'`, « not
  supported on native Windows ») — état assumé, à conserver.
- `uv 0.12.3` présent (`~/.local/bin`) mais **non géré** (hors
  `config.json`/`installer.py`).
- oss-cad-suite vend son propre env (`lib/python3.11`, eggs `py3.11-linux`) :
  `cocotb 2.1.0.dev0`, `pytest 9`, **`migen 0.9.2`**, sans `litex`/`litedram`,
  sans interpréteur (python système via `PYTHONHOME`, script `environment`).
- PyPI officiel : `migen==0.9.2`, `litex==2024.12`, `litedram==2024.12`
  (le 0.9.2 == celui vendu par la suite : cohérent des deux côtés).
- CI (`.github/workflows/ci-simulations.yml`) : shims `pyenv` en tête de
  PATH + `PYTHON_TARGET=3.12` + bootstrap `.venv` « from the GLOBAL
  interpreter ». Sur runners self-hosted avec 3.11 dans `/usr/bin` : trois
  interpréteurs empilés (shims, `/usr/bin`, `.venv`).

### 1.2 Résolution d'interpréteur (la douleur cocotb)
- `python_runner.py:34` : fallback `python3.12`/`python3`/`python` nus du
  PATH si pas de `.venv` — peut rendre silencieusement un autre
  interpréteur (shims pyenv vs `/usr/bin`).
- `build_runner.py:229` : liste en dur `["python3.11", "python3.10",
  "python3.12"]` pour localiser apycula.
- Logs `out/python_reports` : cocotb 1.9.2 tourne depuis
  `.venv/lib/python3.12` mais linke et compile (`perl`, `g++`) contre le
  Verilator de la suite, avec rpath vers les libs du venv — chaque étape
  (build, link, runtime `libpython`) peut résoudre un python différent, et
  rien ne le vérifie. C'est le mécanisme du bug, pas la version 3.12
  (cocotb 1.9.x la supporte, et le bug pré-existait).
- Usage cocotb du repo (`test`, `start_soon`, `triggers`, `clock`) existe
  dans 1.9.2 comme 2.x : **pas de migration 2.x**, on garde 1.9.2.

### 1.3 Patchs toolchain codés en dur (`cli/spinalml_cli/build_runner.py`)
- `ensure_apycula_patched()` (:224) : patch par remplacement de chaîne du
  `gowin_pack.py` de la suite (bug offsets B-multiplier), écrit **dans
  TOOLS_DIR**, avec liste de versions en dur (`python3.11/10/12` — casse à
  3.13+, et mute un outil tiers sans traçabilité de version).
- `patch_gowin_cin_from_logic()` (:254) : backport du commit nextpnr
  `6030081a15`, mute le `pnr.json` routé en place. Porte déjà un
  `TODO(tech-debt)` demandant sa relocalisation (voir
  `docs/bugs/2026-09-gowin-rne-dsp-lut-saturation.md` §8).
- `adapt_constraints_for_ports()` (:298) : pins Tang Primer 20K en dur
  (H11/T10/T13/M11) + heuristiques de noms de ports, au lieu de données
  `boards/`.
- Points sains à ne pas casser : `synth_cmd` (:565) et `pnr_tool` (:642)
  viennent déjà de la config board — le pattern à généraliser.

### 1.4 Leftovers debug (PR nettoyage promise)
- `spinalML/test/src/spinalML/nn/ConvReplicaSpillTest.scala` : tests TMP +
  flags `debug=true`.
- `spinalML/test/src/spinalML/nn/ReplicaSpillE2E.scala` : traces `DBG`
  (derrière flag, défaut off — à conserver comme infra de debug).

## 2. Décisions verrouillées (rappel, non négociable sans re-discussion)

1. **3 environnements, pas un** (décision du 22/09, remplace le « single
   venv » initial) :
   - **Env CLI** (`cli/requirements.txt`, strict minimum : typer, requests,
     rich) — fait tourner la CLI, rien de plus. Rapide à installer partout,
     toutes plateformes.
   - **Env dev** (`/requirements.txt` racine, comme aujourd'hui + nouveaux
     packages) — développement : CLI + tests HW (pytest, cocotb, numpy,
     cocotb-test…) + libs LiteX (`requirements-litex.txt` inclus).
     Installé par **`setup --dev` (conservé)** : cocotb reste opt-in car
     c'est lui le coût (Linux-only par markers déjà en place,
     sensible aux versions, lent). Le mettre par défaut ferait payer le
     setup sim à tout le monde, y compris CLI-only et Windows qui n'en
     ont aucun usage. Sur Windows, `--dev` dégrade gracieusement (les
     markers excluent cocotb, aucun code spécial).
   - **Env portable utilisateur** (`~/.spinalml_tools/pyenv`, géré par
     l'installer comme oss-cad-suite/mill) — runtime portable pour
     l'utilisateur final et l'exe figé : **mêmes pins que dev**
     (une seule source de vérité, emplacement managé), pour que
     `test-all-python` se comporte pareil partout. Jamais touché à la
     main, recréé par `setup`.
   - `setup` seul = env CLI (+ env portable si distribué) ; `setup --dev`
     ajoute l'env dev complet. CI : jobs build → `setup`, jobs sim → 
     `setup --dev`.
2. **`uv 0.12.18` pinné** dans `config.json` (5 plateformes, même pattern
   que oss-cad-suite) + `install_uv()` + manifest. Tag vérifié existant ;
   `spinalml uv ...` exposé en accès direct comme verilator/mill.
3. **`requirements.txt` gardés** + `requirements-litex.txt` nouveau
   (`migen==0.9.2`, `litex==2024.12`, `litedram==2024.12`), PyPI + `==`.
   Pas de vendoring tarball, pas de `--require-hashes` pour l'instant.
4. **Exe PyInstaller = CLI-only**, inchangé. Pas de LiteDRAM dedans
   (itération lente, hidden-imports, double chemin ; les devs `dram-gen`
   ont déjà toute la toolchain de toute façon).
5. **Regen LiteDRAM transparente dans `build`** par hash
   (config+tag vs artefact) : match → `.v` direct, mismatch → regen via le
   pyenv managé + message loud. Jamais de génération périmée silencieuse,
   jamais d'env à gérer à la main.
6. **`.v` généré = lecture seule** (`// GENERATED — DO NOT EDIT` + check
   `regen == committed` à terme). Customisation = paramètres de génération
   + wrapper à nous, jamais d'édit manuel.
7. **Interdit** : symlink `/usr/bin` (sudo, casse l'OS), shims PATH maison
   (pyenv-bis, ne couvre ni les chemins absolus, ni `find_libpython`, ni le
   linker, ni `PYTHONHOME`), toucher au python système.
8. **`setup --dev`** : env sim complet via **`uv venv -p 3.12 --clear`
   + `uv pip install -r`** (pas `pyenv`, pas de `.python-version` : `-p`
   explicite partout, versionnée dans le code ; pas `uv pip sync` :
   il n'installe que le set listé et droppe les transitifs — l'exactitude
   vient de `--clear`). 3.12 par défaut ;
   bascule 3.11 en une ligne si le spike l'exige (voir §5).
   `find_python_interpreter` ne connaît plus aucun nom nu du PATH :
   dev → user → erreur loud. `setup_tool_env` sanitize `PYTHONHOME`.

## 3. Plan d'exécution

### Phase 1 — Environnement (d'abord)
1. `config.json` : entrée `uv` 4 plateformes (+ `requirements-litex.txt`
   créé avec les 3 pins).
2. `installer.py` : `install_uv()` + `install_python_libs()` (structure ;
   contenu LiteDRAM — épinglé, pas encore utilisé).
3. `setup` / `setup --dev` :
   - `setup` : tools (dont uv) + env CLI (`uv venv` + sync
     `cli/requirements.txt`) + env portable si distribué.
   - `setup --dev` : + env dev (`.venv` racine recréé proprement via
     `uv venv` — un venv ne contient aucune donnée utilisateur, la
     recréation est assumée — + sync `/requirements.txt` et
     `requirements-litex.txt`).
   - Les deux terminent par `doctor` (imports + versions).
4. `pylibs.py` + `spinalml pylibs` (chemins par env, installées vs
   pinnées, import check) = « retrouver les libs depuis le CLI ».
5. `python_runner.py:152` : message d'erreur `pip install` → `uv pip sync`.
6. Spike cause racine cocotb : reproduire en env dev `uv` propre à pins
   exacts ; si 3.12 innocenté (attendu), on reste ; sinon bascule 3.11
   via `uv python install` (déjà dispo localement).

### Phase 2 — Nettoyage (PR promise, sur env sain)
7. `ConvReplicaSpillTest` : TMP + flags `debug` dehors (trace derrière
   flag conservée), grep général des leftovers, gate derrière.
8. Docs : `docs/python-env.md` (quel venv, sync, ajout d'une dep,
   politique Windows/Linux, règle `PYTHONHOME` — voir §4).

### Phase 3 — LiteDRAM (après santé, rappelé ici pour l'ordre)
9. Smoke : import + génération d'un core jouet via `examples/` upstream.
10. S1 chip DDR3 exact + faisabilité, S2 modèle sim vs suite bit-exacte,
    puis `dram-gen` + blackbox + `build` (§5 du plan DDR).

### Phase 4 — Refactor patchs toolchain (détaillé §4, avec la Phase 1/2)
11. Module `cli/spinalml_cli/patches/` : `ensure_apycula_patched` +
    `patch_gowin_cin_from_logic` relocalisés, chacun avec prédicat de
    version (suite/manifest) et mode dry-run loggé ; suppression en un
    lieu quand les outils montent de version.
12. Contraintes par board : pins + mapping de noms sortis de
    `adapt_constraints_for_ports` vers `boards/<carte>/` (le code ne garde
    que le moteur d'adaptation générique).
13. Durcissement interpréteur : chemins absolus (`sys.executable` du venv
    propagé), suppression/durcissement des fallbacks nus
    (`python_runner.py:34`, `build_runner.py:229` → découverte via
    `pylibs`/suite, plus liste en dur), sanitize `PATH`/`PYTHONHOME`
    dans les runners.
14. `doctor` anti-fuite : `sys.executable` ∈ venv ? `cocotb.__file__` ∈
    venv ? `libpython` cohérente ? Échec loud à t+0 au lieu d'un bug
    bizarre à t+2h.
15. CI : migrer le bootstrap pyenv vers `uv python install` (mêmes bits
    partout, sans shims ni compilation) ; garder les markers Windows.

## 4. Règle d'hygiène Python (à graver dans `docs/python-env.md`)

- Toujours l'interpréteur explicite du venv managé, jamais de nom nu
  dans les chemins critiques.
- Jamais de `PYTHONHOME` hérité dans les shells de test (le script
  `environment` de la suite le pose : ne pas le sourcer côté tests, ou
  sanitize explicite).
- Tout échec d'env échoue loud avec le chemin fautif (`doctor`), jamais
  silencieux.
- Les versions ne se supposent pas, elles se manifestent (pins + manifest
  + check).

## 5. Risques ouverts (non bloquants pour démarrer)

- Compat `migen 0.9.2 ↔ litex/litedram 2024.12` : à valider au smoke ( §3.9) ;
  fallback = migen en URL git pinnée sur SHA, rien de plus.
- Cause racine cocotb : spike §3.6 ; si elle touche `cocotb-test`/VPI,
  le `doctor` la transformera au pire en échec propre et reproductible.
- Windows : markers gardés ; `uv pip` LiteDRAM = pur Python, même classe
  de risque que typer (déjà installé) ; Gowin EDA reste de toute façon
  Windows-first côté toolchain.
