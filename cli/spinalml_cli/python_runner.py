# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Compatibility shim: the cocotb runner now lives in runners/cocotb.py
(shared engine in runners/engine.py). Names and signatures are unchanged."""

from .runners.cocotb import (
    discover_python_tests,
    find_python_interpreter,
    preflight_test_interpreter,
    run_all_python_tests,
)

__all__ = [
    "discover_python_tests",
    "find_python_interpreter",
    "preflight_test_interpreter",
    "run_all_python_tests",
]
