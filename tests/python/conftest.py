# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import sys
import random

from utils.math_metrics import clear_math_log


def pytest_addoption(parser):
    parser.addoption(
        "--debug-math", 
        action="store_true", 
        default=False, 
        help="Generate true_math_errors.log with detailed precision errors"
    )


def pytest_sessionstart(session):
    seed = int(os.environ.get("SPINALML_SEED", "42"))
    os.environ["SPINALML_SEED"] = str(seed)
    os.environ["RANDOM_SEED"] = str(seed)
    random.seed(seed)
    if session.config.getoption("--debug-math"):
        os.environ["DEBUG_MATH"] = "1"
        clear_math_log()

    if sys.platform == "win32":
        import warnings
        warnings.warn(
            "\n[WARNING] Cocotb + Verilator co-simulations require a Linux environment (Linux, Radxa ARM64, or Windows via WSL).\n"
            "Cocotb's official VPI bridge does not support Verilator on native Windows.\n"
            "To run these tests on Windows, please use WSL: 'wsl python cli/main.py test-all-python'.\n",
            UserWarning
        )

    # Automatically ensure ~/.spinalml_tools and OSS CAD Suite are on PATH & VERILATOR_ROOT is set
    tools_dir = os.path.expanduser("~/.spinalml_tools")
    if os.path.isdir(tools_dir):
        oss_bin = os.path.join(tools_dir, "oss-cad-suite", "bin")
        oss_lib = os.path.join(tools_dir, "oss-cad-suite", "lib")
        verilator_root = os.path.join(tools_dir, "oss-cad-suite", "share", "verilator")

        new_paths = []
        if os.path.isdir(tools_dir):
            new_paths.append(tools_dir)
        if os.path.isdir(oss_bin):
            new_paths.append(oss_bin)
        if os.path.isdir(oss_lib):
            new_paths.append(oss_lib)

        if new_paths:
            existing = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(new_paths) + os.pathsep + existing

        if "VERILATOR_ROOT" not in os.environ and os.path.isdir(verilator_root):
            os.environ["VERILATOR_ROOT"] = verilator_root

