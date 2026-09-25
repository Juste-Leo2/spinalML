# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Managed Python environments powered by uv.

Two environments, one source of truth for pins:
- CLI env     (<root>/.venv)          : strict minimum to run the CLI (/requirements.txt).
- Managed env (TOOLS_DIR/.venv)       : every Python flow (pytest, dram-gen, ...).
  Base (requirements/base.txt) + DRAM (requirements/dram.txt) by default,
  + co-sim extras (requirements/dev.txt, cocotb, Linux-only) with --dev.

Single test runtime rule: test/dram flows ALWAYS spawn the managed env python
by absolute path, from sources and frozen alike. The root .venv never runs tests.

All venvs are created with `uv venv -p 3.12 --clear` (explicit -p, no
.python-version magic) then filled with `uv pip install -r` (full closure
resolved by uv). Exactness comes from --clear: install into an empty venv
leaves no stale extras. (`uv pip sync` is deliberately NOT used: it installs
only the literally-listed set and drops transitive deps like pluggy.)
A venv holds no user data: recreation is always safe.
Plain `setup` is stateless: it always reinstalls the managed env WITHOUT dev
extras (use `setup --dev` to add cocotb back).
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import TOOLS_DIR, get_bin_path, get_project_root, IS_FROZEN

UV_PYTHON = "3.12"


def cli_env_dir() -> Path:
    """Root .venv: CLI runtime only."""
    return get_project_root() / ".venv"


def managed_env_dir() -> Path:
    """TOOLS_DIR/.venv: every Python flow (tests, dram-gen)."""
    return TOOLS_DIR / ".venv"


def venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def cli_requirements() -> List[Path]:
    return [get_project_root() / "requirements.txt"]


def managed_requirements(with_dev: bool = False) -> List[Path]:
    reqdir = get_project_root() / "requirements"
    reqs = [reqdir / "base.txt", reqdir / "dram.txt"]
    if with_dev:
        reqs.append(reqdir / "dev.txt")
    return reqs


def dev_requirements() -> List[Path]:
    return [get_project_root() / "requirements" / "dev.txt"]


def _uv_bin() -> str:
    p = get_bin_path("uv")
    if not p.exists():
        raise RuntimeError(
            f"uv is not installed at {p}. "
            "Run 'spinalml setup' first (uv is installed before any env)."
        )
    return str(p)


def _run(cmd: List[str], label: str, debug: bool = False, console=None) -> None:
    if debug:
        print(f"$ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=not debug, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip().splitlines()
        tail = "\n".join(err[-8:]) if err else "no output"
        raise RuntimeError(f"{label} failed:\n{tail}")


def ensure_env(env_dir: Path, req_files: List[Path], label: str, kind: str,
               debug: bool = False, console=None) -> Path:
    """(Re)creates a venv with uv and installs exact pins. Returns its python."""
    for req in req_files:
        if not req.exists():
            raise RuntimeError(f"{label}: requirements file not found: {req}")
    uv = _uv_bin()
    msg = f"Setting up {label} env at {env_dir} (uv venv -p {UV_PYTHON} + pip install)..."
    if debug:
        print(msg)
    elif console:
        console.print(f"[cyan]{msg}[/cyan]")
    _run([uv, "venv", "-p", UV_PYTHON, "--clear", str(env_dir)],
         f"{label} env creation", debug=debug, console=console)
    py = venv_python(env_dir)
    cmd = [uv, "pip", "install", "--python", str(py)]
    for r in req_files:
        cmd += ["-r", str(r)]
    _run(cmd, f"{label} env install", debug=debug, console=console)
    ok, detail = check_env(py, kind=kind)
    if not ok:
        raise RuntimeError(f"{label} env verification failed: {detail}")
    done = f"{label} env ready: {py}"
    if debug:
        print(done)
    elif console:
        console.print(f"[green]{done}[/green]")
    return py


def ensure_cli_env(debug: bool = False, console=None) -> Path:
    """Ensures the root .venv (CLI runtime) exists and is satisfied.

    If running from inside this environment (e.g. on Windows), avoids --clear
    to prevent OS permission errors on open/locked executables.
    """
    env_dir = cli_env_dir()
    is_running_in_env = False
    try:
        is_running_in_env = Path(sys.executable).resolve().is_relative_to(env_dir.resolve())
    except Exception:
        pass
    if is_running_in_env:
        py = venv_python(env_dir)
        uv = _uv_bin()
        cmd = [uv, "pip", "install", "--python", str(py)]
        for r in cli_requirements():
            cmd += ["-r", str(r)]
        _run(cmd, "CLI env install", debug=debug, console=console)
        return py
    return ensure_env(env_dir, cli_requirements(), "CLI", "cli", debug=debug, console=console)


def ensure_managed_env(with_dev: bool = False, debug: bool = False, console=None) -> Path:
    extra = " + dev extras" if with_dev else " (no dev extras; pass --dev for cocotb)"
    return ensure_env(managed_env_dir(), managed_requirements(with_dev),
                      f"managed{extra}", "full" if with_dev else "managed",
                      debug=debug, console=console)


def check_env(python: Path, kind: str) -> Tuple[bool, str]:
    """Verifies a venv python: exists, is 3.12, uses its own prefix, imports key modules."""
    if not python.exists():
        return False, f"{python} does not exist"
    mods = "typer,rich,requests" if kind == "cli" else "pytest,numpy"
    if kind == "full" and os.name != "nt":
        mods += ",cocotb"
    # importlib.util.find_spec avoids executing module code (fast, side-effect free).
    code = (
        "import sys, importlib.util; "
        f"mods='{mods}'.split(','); "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(sys.version.split()[0] + '|' + sys.prefix + '|' + ','.join(missing)); "
    )
    try:
        res = subprocess.run([str(python), "-c", code],
                             capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"could not execute {python}: {e}"
    if res.returncode != 0:
        return False, f"{python} -c failed: {(res.stderr or '').strip()[-200:]}"
    try:
        version, prefix, missing = res.stdout.strip().split("|")
    except ValueError:
        return False, f"unexpected check output: {res.stdout.strip()[-200:]}"
    if not version.startswith(UV_PYTHON + "."):
        return False, f"python {version}, expected {UV_PYTHON}.x"
    if Path(prefix).resolve() != python.parent.parent.resolve():
        return False, f"prefix leak: {prefix} (interpreter: {python})"
    if missing:
        return False, f"missing modules: {missing}"
    return True, f"python {version}, {len(mods.split(','))} modules ok"


def has_dev_extras(python: Path) -> bool:
    """True iff the managed env has the cocotb set (i.e. setup --dev was used)."""
    code = "import importlib.util; print(importlib.util.find_spec('cocotb') is not None)"
    try:
        res = subprocess.run([str(python), "-c", code],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    return res.returncode == 0 and res.stdout.strip() == "True"


def doctor_data() -> List[Dict[str, str]]:
    """Per-env health rows for `spinalml doctor` / end-of-setup report."""
    rows = []
    if not IS_FROZEN:
        rows.append(("CLI", cli_env_dir(), "cli"))
    rows.append(("managed", managed_env_dir(), "managed"))

    out = []
    for name, path, kind in rows:
        ok, detail = check_env(venv_python(path), kind)
        out.append({"env": name, "path": str(path),
                    "status": "OK" if ok else "FAIL", "detail": detail})
    managed_py = venv_python(managed_env_dir())
    if managed_py.exists():
        dev = has_dev_extras(managed_py)
        out.append({"env": "dev-extras", "path": "cocotb set in managed env",
                    "status": "installed" if dev else "missing",
                    "detail": "setup --dev installs it; plain setup removes it"})
    exe = Path(sys.executable)
    if IS_FROZEN:
        home = "frozen"
    else:
        managed = [cli_env_dir(), managed_env_dir()]
        home = "managed" if any(exe.is_relative_to(m) for m in managed if m.exists()) else "EXTERNAL"
    out.append({"env": "cli-process", "path": str(exe),
                "status": home, "detail": "interpreter running this CLI"})
    return out


def print_doctor(debug: bool = False, console=None) -> bool:
    """Prints the env health table. Returns True iff all managed envs are OK."""
    rows = doctor_data()
    ok = all(r["status"] in ("OK", "installed", "managed", "frozen", "EXTERNAL")
             for r in rows if r["env"] not in ("cli-process", "dev-extras"))
    if debug or console is None:
        for r in rows:
            print(f"[{r['status']}] {r['env']:12s} {r['path']} :: {r['detail']}")
    else:
        from rich.table import Table
        table = Table(title="Python environments (uv-managed)", border_style="blue")
        table.add_column("Env", style="bold")
        table.add_column("Path")
        table.add_column("Status")
        table.add_column("Detail", style="dim")
        for r in rows:
            color = {"OK": "green", "installed": "green", "managed": "green",
                     "EXTERNAL": "yellow", "missing": "yellow"}.get(r["status"], "red")
            table.add_row(r["env"], r["path"], f"[{color}]{r['status']}[/{color}]", r["detail"])
        console.print(table)
    if not ok:
        hint = "Run 'spinalml setup' (or 'spinalml setup --dev' for co-simulation) to (re)create them."
        if debug or console is None:
            print(hint)
        else:
            console.print(f"[bold red]{hint}[/]")
    return ok


_PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(==|>=|~=|!=|<=|>|<|===)\s*([^;\s#]+)")


def collect_pins(req_files: List[Path], _seen: Optional[set] = None) -> Dict[str, str]:
    """Parses == pins from requirements files, following -r includes (first pin wins)."""
    seen = _seen if _seen is not None else set()
    pins: Dict[str, str] = {}
    for req in req_files:
        req = req.resolve()
        if req in seen or not req.exists():
            continue
        seen.add(req)
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                inc = (req.parent / line[3:].strip()).resolve()
                for k, v in collect_pins([inc], seen).items():
                    pins.setdefault(k, v)
                continue
            m = _PIN_RE.match(line)
            if m:
                pins.setdefault(m.group(1).lower(), m.group(2) + m.group(3))
    return pins


def installed_version(python: Path, package: str) -> Optional[str]:
    code = (
        "import importlib.metadata as m; "
        f"print(m.version('{package}'))"
    )
    try:
        res = subprocess.run([str(python), "-c", code],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def pylibs_data() -> List[Dict[str, str]]:
    """Per-env rows: pinned vs installed versions for `spinalml pylibs`."""
    envs = []
    if not IS_FROZEN and (get_project_root() / "requirements.txt").exists():
        envs.append(("CLI", cli_env_dir(), cli_requirements()))
    envs.append(("managed", managed_env_dir(),
                 managed_requirements() + dev_requirements()))
    rows = []
    for name, path, reqs in envs:
        py = venv_python(path)
        pins = collect_pins(reqs)
        if not py.exists():
            for pkg, pin in sorted(pins.items()):
                rows.append({"env": name, "package": pkg, "pinned": pin,
                             "installed": "-", "status": "NO ENV"})
            continue
        for pkg, pin in sorted(pins.items()):
            ver = installed_version(py, pkg)
            if ver is None:
                status = "MISSING"
            elif pin.startswith("==") and ver != pin[2:]:
                status = "DRIFT"
            else:
                status = "OK"
            rows.append({"env": name, "package": pkg, "pinned": pin,
                         "installed": ver or "-", "status": status})
    return rows


def print_pylibs(debug: bool = False, console=None) -> bool:
    """Prints pinned-vs-installed table. Returns True iff no MISSING/DRIFT/NO ENV,
    except dev extras which are allowed to be MISSING without --dev."""
    rows = pylibs_data()
    dev_pkgs = set(collect_pins(dev_requirements()))
    ok = all(r["status"] == "OK" or (r["env"] == "managed" and r["package"] in dev_pkgs)
             for r in rows)
    if debug or console is None:
        for r in rows:
            print(f"[{r['status']}] {r['env']:8s} {r['package']:15s} pinned={r['pinned']:10s} installed={r['installed']}")
    else:
        from rich.table import Table
        table = Table(title="Python libs: pinned vs installed", border_style="blue")
        table.add_column("Env", style="bold")
        table.add_column("Package")
        table.add_column("Pinned")
        table.add_column("Installed")
        table.add_column("Status")
        for r in rows:
            color = {"OK": "green", "DRIFT": "yellow"}.get(r["status"], "red")
            if r["status"] == "MISSING" and r["env"] == "managed" and r["package"] in dev_pkgs:
                color = "yellow"
            table.add_row(r["env"], r["package"], r["pinned"], r["installed"],
                          f"[{color}]{r['status']}[/{color}]")
        console.print(table)
        if any(r["env"] == "managed" and r["package"] in dev_pkgs and r["status"] == "MISSING" for r in rows):
            console.print("[yellow]dev extras (cocotb) missing: `spinalml setup --dev` to install.[/yellow]")
    return ok
