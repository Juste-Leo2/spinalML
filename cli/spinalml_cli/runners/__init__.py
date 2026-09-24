# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Sequential test runners (shared engine + thin per-flow modules)."""

from .engine import (
    AbortItem,
    RunContext,
    RunnerHooks,
    RunnerSpec,
    check_tool_or_report,
    run_sequential,
    ABORT,
    ERROR,
    FAIL,
    PASS,
    SKIP,
    TIMEOUT,
    RETRY_DELAY_S,
)

__all__ = [
    "AbortItem",
    "RunContext",
    "RunnerHooks",
    "RunnerSpec",
    "check_tool_or_report",
    "run_sequential",
    "ABORT",
    "ERROR",
    "FAIL",
    "PASS",
    "SKIP",
    "TIMEOUT",
    "RETRY_DELAY_S",
]
