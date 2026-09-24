# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Compatibility shim: the formal runner now lives in runners/formal.py
(shared engine in runners/engine.py). Names and signatures are unchanged."""

from .runners.formal import discover_formal_specs, run_all_formal_tests

__all__ = ["discover_formal_specs", "run_all_formal_tests"]
