# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""`spinalml dram-gen`: LiteDRAM core generation (thin CLI wrapper).

This module must NEVER import litex/migen/litedram: it runs inside the CLI
env (root .venv, typer/rich/requests only). Generation itself is delegated
to `dram/gen/gowin_gen.py`, executed with the managed env python by absolute
path (same single-runtime rule as the cocotb runner).
"""

import subprocess
from pathlib import Path
from typing import Optional

from .config import get_project_root
from .pyenv import managed_env_dir, venv_python


def dram_config_for_board(board: str) -> Path:
    """Returns dram/configs/<board>.yml or raises a loud error."""
    cfg = get_project_root() / "dram" / "configs" / f"{board}.yml"
    if not cfg.is_file():
        raise FileNotFoundError(
            f"No DRAM config for board '{board}' (expected {cfg}). "
            f"Only boards with a dram/configs/<slug>.yml can use --dram."
        )
    return cfg


def dram_out_dir() -> Path:
    return get_project_root() / "dram" / "out"


def dram_prims() -> Path:
    return get_project_root() / "dram" / "prims" / "gowin_bb.v"


def run_dram_gen(board: str, out_dir: Optional[Path] = None,
                 name: str = "litedram_core", console=None,
                 debug: bool = False) -> int:
    """Validates the board config and generates the LiteDRAM core Verilog.

    Returns the generator return code (0 = ok). Raises on missing pieces.
    """
    root = get_project_root()
    gen = root / "dram" / "gen" / "gowin_gen.py"
    validate = root / "dram" / "gen" / "validate.py"
    if not gen.is_file():
        raise RuntimeError(f"DRAM generator not found: {gen}")
    config = dram_config_for_board(board)
    out = out_dir or dram_out_dir()

    py = venv_python(managed_env_dir())
    if not py.exists():
        raise RuntimeError(
            f"Managed python not found at {py}. Run 'spinalml setup' first.")

    def run(cmd, label):
        if debug:
            print(f"$ {' '.join(str(c) for c in cmd)}")
        elif console:
            console.print(f"[cyan]{label}...[/cyan]")
        res = subprocess.run([str(c) for c in cmd], cwd=str(root))
        return res.returncode

    rc = run([py, validate, config], f"[dram-gen] Validating {config.name}")
    if rc != 0:
        return rc
    rc = run([py, gen, "--config", config, "--out", out, "--name", name],
             f"[dram-gen] Generating {name}.v for board '{board}'")
    if rc != 0:
        return rc

    core_v = out / f"{name}.v"
    if not core_v.is_file():
        msg = f"[dram-gen] generator exited 0 but {core_v} is missing."
        if console and not debug:
            console.print(f"[bold red]{msg}[/bold red]")
        else:
            print(msg)
        return 1
    ok_msg = f"[dram-gen] Wrote {core_v} ({core_v.stat().st_size // 1024} KiB)."
    if debug or console is None:
        print(ok_msg)
    elif console:
        console.print(f"[green]{ok_msg}[/green]")
    return 0
