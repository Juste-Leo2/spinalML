# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Shared tool environment for runner subprocesses (moved verbatim from test_runner)."""

import os
from pathlib import Path  # noqa: F401 (kept for import compatibility)
from typing import Dict  # noqa: F401 (kept for import compatibility)

from ..config import TOOLS_DIR, get_bin_path


def setup_tool_env() -> dict:
    """Prepares environment with verilator and compiler toolchains on PATH.

    Sanitizes PYTHONHOME: the oss-cad-suite `environment` script sets it to the
    suite's vendored python, which would hijack stdlib resolution of every
    subprocess python (the cocotb env bug class). Managed venvs must resolve
    their own stdlib.
    """
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)

    oss_bin = TOOLS_DIR / "oss-cad-suite" / "bin"
    oss_lib = TOOLS_DIR / "oss-cad-suite" / "lib"
    w64_bin = TOOLS_DIR / "w64devkit" / "bin"
    verilator_root = TOOLS_DIR / "oss-cad-suite" / "share" / "verilator"

    new_paths = [str(oss_bin), str(oss_lib)]
    if verilator_root.exists():
        env["VERILATOR_ROOT"] = str(verilator_root)
        verilator_bin = verilator_root / "bin"
        if verilator_bin.exists():
            new_paths.append(str(verilator_bin))

    if w64_bin.exists():
        new_paths.append(str(w64_bin))

    mill_bin = get_bin_path("mill")
    if mill_bin.exists() and str(mill_bin.parent) not in new_paths:
        new_paths.insert(0, str(mill_bin.parent))

    existing_path = env.get("PATH", "") or env.get("Path", "")
    combined_path = os.pathsep.join(new_paths) + os.pathsep + existing_path
    env["PATH"] = combined_path
    env["Path"] = combined_path
    return env
