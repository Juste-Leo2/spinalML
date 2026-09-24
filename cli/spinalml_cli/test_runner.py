# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Compatibility shim: the ScalaTest runner now lives in runners/scala.py
(shared engine in runners/engine.py). Names and signatures are unchanged."""

from .runners.env import setup_tool_env
from .runners.scala import discover_tests, run_all_tests

__all__ = ["setup_tool_env", "discover_tests", "run_all_tests"]
