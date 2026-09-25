# Python environments (uv-managed)

> Single source of truth for pins: `requirements/` + `/requirements.txt`.
> Humans never hand-install beyond bootstrapping the CLI: `spinalml setup [--dev]`.

## 1. Layout — 2 envs, no more

| Env | Path | Content | Created by |
| --- | ---- | ------- | ---------- |
| CLI | `<root>/.venv` | `/requirements.txt` (typer, requests, rich) | Dev bootstrap (`uv venv` + `uv pip install`) |
| managed | `~/.spinalml_tools/.venv` | base + dram (LiteDRAM), + dev extras with `--dev` | `setup` / `setup --dev` |

Requirement sets (`requirements/`):

| File | Packages | Installed… |
| ---- | -------- | ---------- |
| `base.txt` | pytest, pytest-cov, numpy (`==`) | …by plain `setup` |
| `dram.txt` | migen, litex, litedram (`==`, PyPI) | …by plain `setup` (pure Python, cheap) |
| `dev.txt` | cocotb, cocotb-test, cocotbext-axi (`==`, `sys_platform != 'win32'`) | …by `setup --dev` only |

`/requirements.txt` (repo root) is the CLI set only. It keeps the universal
`pip install -r requirements.txt` reflex working with a safe, minimal result.

**Single-runtime rule.** Test and generation flows (`test-all-python`,
future `dram-gen`) ALWAYS spawn the managed env python by absolute path —
from sources and frozen alike. The root `.venv` runs the CLI only and can
never run tests. Shell activated on `.venv` + `pytest` by hand does not work:
use `spinalml test-all-python`. This is on purpose.

## 2. Flows

```bash
# Bootstrap from a fresh clone (uv auto-provisions Python 3.12, explicit -p, no magic file)
uv venv -p 3.12 --clear .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt   # CLI only

python cli/main.py setup             # tools + managed env (base + dram / LiteDRAM ready)
python cli/main.py setup --dev       # + cocotb (co-simulation)

python cli/main.py doctor            # health: interpreter, pins, leaks
python cli/main.py pylibs            # pinned vs installed per env
python cli/main.py uv --version      # managed uv (0.12.18 pinné)
```

`setup` is **stateless for the managed env**: plain `setup` always reinstalls the managed env
WITHOUT dev extras (it `--clear`s, cocotb included). There is no sticky flag:
pass `--dev` every time you need co-sim. CI therefore runs `setup --dev`
uniformly in every job — a plain `setup` interleaved between jobs sharing one
runner would wipe cocotb for the next job. `setup` does NOT `--clear` the root `.venv`,
avoiding executable locking errors on Windows and ensuring clean separation.

## 3. Adding a dependency

1. Add the pin to the right file (`base` / `dram` / `dev.txt`; `==`, never `>=`,
   except the CLI set which intentionally uses ranges).
2. Re-run `setup` / `setup --dev` (venvs are `--clear`ed: exactness guaranteed).
3. Check `spinalml pylibs` (must be all OK) and `spinalml doctor`.

## 4. Platform policy

- cocotb is excluded on native Windows by markers (`sys_platform != 'win32'`,
  kept as-is): `--dev` degrades gracefully there, no special code. Co-sim runs
  on Linux / WSL (see the `test-all-python` notice).
- The LiteX stack (`migen`, `litex`, `litedram`) is pure Python: same risk class as typer, installs everywhere.
- The PyInstaller exe is CLI-only (typer/rich/requests). Heavy sets (LiteDRAM, pytest) live in
  the managed env (`~/.spinalml_tools/.venv`), never in the exe — that is what keeps the binary small,
  avoids embedding complex dependencies, and ensures the end-user only needs the standalone binary
  and the single managed environment created by `spinalml setup`.

## 5. Hygiene rules (never break these)

- Explicit managed interpreters by absolute path in all critical code paths.
  Never a bare `python3`/`python` from PATH (a silent wrong interpreter is
  worse than a loud error).
- Never inherit `PYTHONHOME` into test subprocesses (the oss-cad-suite
  `environment` script sets it to the suite's vendored python).
- Env failures are loud with the offending path (`doctor`), never silent.
- Versions are manifested (pins + installer manifest + checks), never assumed.

## 6. Lessons learned (why these rules exist)

1. **The cocotb bug was resolution, never version.** Three interpreters piled
   up (pyenv shims, `/usr/bin`, `.venv`); bare-PATH fallbacks (`shutil.which`)
   silently picked the wrong one; each stage (build, link, runtime libpython)
   could resolve differently. Proven by poisoning: a fake `python3` shim first
   in PATH is picked by the old code and ignored by the new resolver; 3.12 +
   cocotb 1.9.2 pass clean in a fresh uv env. Do not "fix" this by changing
   Python versions.
2. **`uv pip sync` drops transitive deps.** It installs only the literally
   listed set (`pluggy`, `annotated_doc` missing → dead CLI). We use
   `uv venv --clear` + `uv pip install -r` instead: exactness comes from the
   emptied venv, closure from the resolver.
3. **`sim_build/` caches absolute venv paths.** After any env move/rebuild,
   `make` fails with `No rule to make target '.../.venv/.../verilator.cpp'`.
   Fix: `rm -rf sim_build` (gitignored build artifact). Always wipe it when
   envs move.
4. **`PYTHONHOME` is lethal.** Pointed at the suite's vendored python, the
   interpreter cannot even boot (`No module named 'encodings'`). `setup_tool_env()`
   strips it from every test subprocess env; never source the suite's
   `environment` script on the test side.
5. **CDN blocks urllib's default User-Agent.** `releases.astral.sh` answers
   403 to `Python-urllib/x.y` (curl works). `download_file()` sends a neutral
   browser UA for all tools.
6. **No `.python-version` magic.** The old file said `3.12.1` while 3.12.3 was
   in use: obsolete and contradictory. Every `uv` call passes `-p 3.12`
   explicitly (one-line switch to 3.11 if ever needed).
7. **Ordre des contraintes CST sans effet.** `pins.cst` adapté suit l'ordre
   des ports du top, pas celui du `.cst` board : nextpnr le consomme comme un
   set. Seul le contenu compte (prouvé byte-identique à l'ancien sur les deux
   nommages).
8. **Bitstreams move with RTL.** `examples/Mnist/top.fs` went stale across
   PRs #10/#11 (8 RTL files changed, LUT 10 219 → 10 265) without refresh.
   Any PR touching `spinalML/src` must rebuild + refresh `top.fs` (or record
   why not).
