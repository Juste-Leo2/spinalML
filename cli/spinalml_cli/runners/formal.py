# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""SymbiYosys formal runner (test-all-formal): discovery + hooks, loop in engine."""

import re
import subprocess
from pathlib import Path
from typing import List, Optional

from ..config import get_active_framework_root, get_bin_path
from .engine import (
    RunContext,
    RunnerHooks,
    RunnerSpec,
    check_tool_or_report,
    run_sequential,
)


def discover_formal_specs(formal_src_dir: Path, filter_pattern: Optional[str] = None) -> List[str]:
    """Finds all concrete formal verification objects in symbolicTest/."""
    specs = []
    if not formal_src_dir.exists():
        return specs

    for file_path in sorted(formal_src_dir.rglob("*Formal.scala")):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
        pkg = pkg_match.group(1) if pkg_match else ""

        # Look for object definition with main or doVerify
        obj_matches = re.findall(r'^\s*object\s+(\w+)', content, re.MULTILINE)
        if obj_matches:
            # Prefer object matching filename or containing 'Formal'
            target_obj = None
            stem = file_path.stem
            for obj in obj_matches:
                if obj.lower() == stem.lower():
                    target_obj = obj
                    break
            if not target_obj:
                for obj in obj_matches:
                    if "formal" in obj.lower():
                        target_obj = obj
                        break
            if not target_obj:
                target_obj = obj_matches[0]

            fqcn = f"{pkg}.{target_obj}" if pkg else target_obj

            if filter_pattern:
                if re.search(filter_pattern, fqcn, re.IGNORECASE):
                    specs.append(fqcn)
            else:
                specs.append(fqcn)

    return sorted(list(set(specs)))


class _FormalHooks(RunnerHooks):
    def __init__(self, mill_bin: str, timeout: int):
        self._mill_bin = mill_bin
        self._timeout = timeout

    def check_tool(self, ctx: RunContext) -> Optional[str]:
        if not Path(self._mill_bin).exists():
            return (f"[bold red]Error:[/] Mill executable not found at {self._mill_bin}. "
                    "Run 'spinalml setup' first.")
        return None

    def tool_error_style(self) -> Optional[str]:
        return "red"

    def build_cmd(self, item: str, ctx: RunContext) -> List[str]:
        return [self._mill_bin, "--no-server", "--disable-ticker",
                "spinalML.test.runMain", item]

    def run_timeout(self, item: str, ctx: RunContext) -> Optional[float]:
        return self._timeout

    def classify(self, res: subprocess.CompletedProcess, item: str, ctx: RunContext) -> str:
        from .engine import FAIL, PASS
        ok = ((res.returncode == 0) and ("Assert failed" not in res.stdout)
              and ("FAIL" not in res.stdout or "SUCCESS" in res.stdout))
        return PASS if ok else FAIL

    def log_header(self, item: str, kind: str, ctx: RunContext) -> str:
        if kind == "timeout":
            return f"=== FORMAL VERIFICATION TIMED OUT ({self._timeout}s): {item} ==="
        return f"=== FORMAL VERIFICATION FAILED: {item} ==="

    def failure_hint(self, output: str) -> Optional[str]:
        if "checksum format error" in output or "scalaCompilerBridge" in output:
            return ("       [bold yellow]↳ Detected Coursier cache checksum corruption.[/bold yellow] "
                    "Run [bold cyan]python cli/main.py clean-cache[/bold cyan] to clear.")
        return None

    def detail_title(self, item: str, kind: str, ctx: RunContext) -> str:
        if kind == "fail":
            return f"Formal Failure Output: {item}"
        return super().detail_title(item, kind, ctx)


def run_all_formal_tests(
    filter_pattern: Optional[str] = None,
    fail_fast: bool = False,
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: int = 900,
    verbose: bool = False,
    ci_sleep: float = 0.0,
    retries: int = 1
) -> int:
    project_root = get_active_framework_root()
    formal_src = project_root / "spinalML" / "test" / "src" / "spinalML" / "symbolicTest"


    if log_dir is None:
        log_dir = project_root / "out" / "formal_reports"
    log_dir.mkdir(parents=True, exist_ok=True)

    spec = RunnerSpec(
        title="[bold cyan]SpinalML Formal Verification Runner (SymbiYosys / BMC)[/]",
        mode_line=f"Mode: [bold]Sequential (1-by-1)[/] | Timeout per suite: [bold]{timeout}s[/]",
        item_noun="formal suites",
        run_verb="Verifying",
        run_verb_color="magenta",
        total_label="Total Formal Suites Discovered",
        exec_label="Suites Executed",
        summary_title="Formal Verification Summary",
        fail_summary_title="Failed Formal Specs Summary:",
        fail_first_col="Formal Spec",
        all_pass_msg="All {n} formal verification suites passed successfully!",
        empty_msg="No formal verification suites found matching criteria.",
        dry_title="Discovered {n} Formal Verification Suites",
        fail_log_prefix="FORMAL VERIFICATION FAILED",
        panel_color="cyan",
    )
    ctx = RunContext(project_root=project_root, log_dir=log_dir,
                     fail_fast=fail_fast, verbose=verbose, ci_sleep=ci_sleep,
                     extra={}, max_retries=retries)
    hooks = _FormalHooks(str(get_bin_path("mill")), timeout)
    if not check_tool_or_report(hooks, ctx):
        return 1

    specs = discover_formal_specs(formal_src, filter_pattern)
    return run_sequential(spec, hooks, specs, ctx, dry_run=dry_run)
