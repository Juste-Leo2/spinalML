# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""ScalaTest sequential runner (test-all): discovery + hooks, loop in engine."""

import re
import subprocess
from pathlib import Path
from typing import List, Optional

from ..config import get_active_framework_root, get_bin_path
from .engine import (
    AbortItem,
    RunContext,
    RunnerHooks,
    RunnerSpec,
    check_tool_or_report,
    run_sequential,
)


def discover_tests(test_src_dir: Path, filter_pattern: Optional[str] = None) -> List[str]:
    """Finds all concrete ScalaTest classes in the test directory."""
    tests = []
    if not test_src_dir.exists():
        return tests

    for file_path in sorted(test_src_dir.rglob("*.scala")):
        # Skip temporary scaffold test files
        if "cli_test_temp" in file_path.parts:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
        pkg = pkg_match.group(1) if pkg_match else ""

        # Match class definitions that extend Suite or FunSuite, but not abstract classes
        for match in re.finditer(r'^\s*(?!abstract\s+)class\s+(\w+)\s+extends\s+.*(?:Suite|FunSuite)', content, re.MULTILINE):
            cls_name = match.group(1)
            fqcn = f"{pkg}.{cls_name}" if pkg else cls_name

            if filter_pattern:
                if re.search(filter_pattern, fqcn, re.IGNORECASE):
                    tests.append(fqcn)
            else:
                tests.append(fqcn)

    return sorted(tests)


_SPEC = RunnerSpec(
    title="[bold green]SpinalML Sequential Test Suite Runner[/]",
    mode_line="Mode: [bold]Sequential (1-by-1)[/] to prevent Verilator/G++ RAM exhaustion on Windows",
    item_noun="test suites",
    run_verb="Running",
    run_verb_color="blue",
    total_label="Total Suites Discovered",
    exec_label="Suites Executed",
    summary_title="Test Execution Summary",
    fail_summary_title="Failed Tests Summary:",
    fail_first_col="Test Class",
    all_pass_msg="All {n} tests passed successfully!",
    empty_msg="No tests found matching criteria.",
    dry_title="Discovered {n} test suites",
    fail_log_prefix="TEST RUN FAILED",
    panel_color="green",
)


class _ScalaHooks(RunnerHooks):
    def __init__(self, mill_bin: str):
        self._mill_bin = mill_bin

    def check_tool(self, ctx: RunContext) -> Optional[str]:
        if not Path(self._mill_bin).exists():
            return (f"[bold red]Error:[/] Mill executable not found at {self._mill_bin}. "
                    "Run 'spinalml setup' first.")
        return None

    def tool_error_style(self) -> Optional[str]:
        return "red"

    def build_cmd(self, item: str, ctx: RunContext) -> List[str]:
        return [self._mill_bin, "--no-server", "--disable-ticker",
                "spinalML.test.testOnly", item]

    def log_header(self, item: str, kind: str, ctx: RunContext) -> str:
        return f"=== TEST RUN FAILED: {item} ==="

    def failure_hint(self, output: str) -> Optional[str]:
        if "checksum format error" in output or "scalaCompilerBridge" in output:
            return ("       [bold yellow]↳ Detected Coursier cache checksum corruption.[/bold yellow] "
                    "Run [bold cyan]python cli/main.py clean-cache[/bold cyan] to clear.")
        return None

    def detail_title(self, item: str, kind: str, ctx: RunContext) -> str:
        return f"Failure Output: {item}" if kind == "fail" else super().detail_title(item, kind, ctx)


def run_all_tests(
    filter_pattern: Optional[str] = None,
    fail_fast: bool = False,
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False,
    ci_sleep: float = 0.0
) -> int:
    project_root = get_active_framework_root()
    test_src = project_root / "spinalML" / "test" / "src"


    if log_dir is None:
        log_dir = project_root / "out" / "test_reports"
    log_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(project_root=project_root, log_dir=log_dir,
                     fail_fast=fail_fast, verbose=verbose, ci_sleep=ci_sleep, extra={})
    hooks = _ScalaHooks(str(get_bin_path("mill")))
    if not check_tool_or_report(hooks, ctx):
        return 1

    tests = discover_tests(test_src, filter_pattern)
    return run_sequential(_SPEC, hooks, tests, ctx, dry_run=dry_run)
