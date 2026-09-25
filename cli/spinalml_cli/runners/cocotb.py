# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Pytest/cocotb runner (test-all-python): discovery + hooks, loop in engine."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from ..config import CLI_DIR, get_project_root
from ..pyenv import managed_env_dir, venv_python
from .engine import (
    AbortItem,
    FAIL,
    PASS,
    SKIP,
    RunContext,
    RunnerHooks,
    RunnerSpec,
    check_tool_or_report,
    run_sequential,
)


def discover_python_tests(test_dir: Path, filter_pattern: Optional[str] = None) -> List[Path]:
    """
    Finds all Python test files (test_*.py) in tests/python/.
    If filter_pattern is given, filters by filename match first, or by test content match.
    """
    if not test_dir.exists():
        return []

    all_files = sorted(test_dir.glob("test_*.py"))
    if not filter_pattern:
        return all_files

    # 1. Try matching filename directly
    matched_by_name = [
        f for f in all_files
        if re.search(filter_pattern, f.stem, re.IGNORECASE) or re.search(filter_pattern, f.name, re.IGNORECASE)
    ]
    if matched_by_name:
        return matched_by_name

    # 2. If no filename match, check file contents (e.g. specific test function or dtype name)
    matched_by_content = []
    for f in all_files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            if re.search(filter_pattern, content, re.IGNORECASE):
                matched_by_content.append(f)
        except Exception:
            continue

    return matched_by_content


def find_python_interpreter() -> Optional[str]:
    """Single test runtime: the uv-managed env by absolute path (sources and frozen alike)."""
    py = venv_python(managed_env_dir())
    return str(py) if py.exists() else None


def preflight_test_interpreter(py_bin: str) -> Optional[str]:
    """Fails loud if py_bin cannot import the test stack. Returns error text or None."""
    mods = "pytest"
    if sys.platform != "win32":
        mods += ",cocotb"
    code = (
        "import importlib.util; "
        f"missing=[m for m in '{mods}'.split(',') if importlib.util.find_spec(m) is None]; "
        "print(','.join(missing)); "
    )
    try:
        res = subprocess.run([py_bin, "-c", code], capture_output=True, text=True, timeout=60)
    except Exception as e:
        return f"could not execute {py_bin}: {e}"
    if res.returncode != 0:
        return f"{py_bin} is broken: {(res.stderr or '').strip()[-300:]}"
    missing = res.stdout.strip()
    if missing:
        return (f"{py_bin} is missing: {missing}. "
                "Run 'spinalml setup' (or 'spinalml setup --dev' for co-simulation).")
    return None


def _test_dir(project_root: Path) -> Path:
    return project_root / "tests" / "python"


class _CocotbHooks(RunnerHooks):
    def prepare_all(self, ctx: RunContext) -> None:
        # DEBUG_MATH: a full CLI run starts from a fresh math log. Each test file
        # is then launched as its own pytest process and appends to it, so lines
        # from previous files must never be truncated mid-run.
        if ctx.extra.get("debug_math"):
            math_log = ctx.project_root / "tests" / "true_math_errors.log"
            try:
                math_log.unlink()
            except OSError:
                pass

        # Prevent VPATH ../verilator.o pollution: ensure no orphan object files exist at sim_build root
        sim_build_root = ctx.project_root / "sim_build"
        if sim_build_root.exists():
            for orphan in sim_build_root.glob("*.o"):
                try:
                    orphan.unlink()
                except OSError:
                    pass

    def build_env(self, ctx: RunContext) -> dict:
        from .env import setup_tool_env
        env = setup_tool_env()
        cli_str = str(CLI_DIR)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cli_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else cli_str
        if ctx.ci_sleep > 0:
            env["SM_CI_SLEEP"] = str(ctx.ci_sleep)
        return env

    def build_cmd(self, item: str, ctx: RunContext) -> List[str]:
        test_file = _test_dir(ctx.project_root) / item
        rel_test_path = str(test_file.relative_to(ctx.project_root))
        py_bin = find_python_interpreter()
        if not py_bin:
            raise AbortItem("Could not locate the managed Python env for pytest/cocotb.\n"
                            "Run 'spinalml setup' (or 'spinalml setup --dev' for co-simulation).")
        preflight_err = preflight_test_interpreter(py_bin)
        if preflight_err:
            raise AbortItem(preflight_err)
        cmd = [py_bin, "-m", "pytest", rel_test_path]

        filter_pattern = ctx.extra.get("filter_pattern")
        # Only pass -k to pytest if the filter was targeting a specific test inside the file (content match, not filename)
        matched_filename = bool(re.search(filter_pattern, test_file.stem, re.IGNORECASE) or re.search(filter_pattern, test_file.name, re.IGNORECASE)) if filter_pattern else False
        if filter_pattern and not matched_filename:
            cmd.extend(["-k", filter_pattern])

        if ctx.fail_fast:
            cmd.append("-x")
        if ctx.extra.get("debug_math"):
            cmd.append("--debug-math")
        extra_args = ctx.extra.get("extra_args")
        if extra_args:
            cmd.extend(extra_args)

        # Run with -s to keep output captured properly
        cmd.append("-s")
        return cmd

    def classify(self, res: subprocess.CompletedProcess, item: str, ctx: RunContext) -> str:
        if res.returncode == 0:
            return PASS
        # Pytest exit code 5: no tests collected matching pattern inside this file
        if res.returncode == 5 and ctx.extra.get("filter_pattern"):
            return SKIP
        return FAIL

    def skip_suffix(self, item: str, ctx: RunContext) -> str:
        return f" (no tests matched '{ctx.extra.get('filter_pattern')}')"

    def log_name(self, item: str, kind: str, ctx: RunContext) -> str:
        stem = Path(item).stem
        return f"{stem}.log" if kind == "fail" else f"{stem}_exception.log"

    def log_header(self, item: str, kind: str, ctx: RunContext) -> str:
        return f"=== PYTHON CO-SIMULATION FAILED: {item} ==="

    def exception_log_content(self, item: str, exc: Exception, cmd: List[str],
                              ctx: RunContext) -> Optional[str]:
        return f"=== EXCEPTION RUNNING {item} ===\n{str(exc)}\n"


_SPEC = RunnerSpec(
    title="[bold blue]SpinalML Python Co-Simulation Suite Runner (Pytest / Cocotb)[/]",
    mode_line="Mode: [bold]Sequential (file-by-file)[/] to prevent RAM/VPI exhaustion on self-hosted runners",
    item_noun="test files",
    run_verb="Running",
    run_verb_color="blue",
    total_label="Total Files Discovered",
    exec_label="Files Executed",
    summary_title="Python Co-Simulation Summary",
    fail_summary_title="Failed Tests Summary:",
    fail_first_col="Test File",
    all_pass_msg="All {n} Python test files passed successfully!",
    empty_msg="No Python co-simulation tests found matching criteria.",
    dry_title="Discovered {n} Python Co-Simulation Test Files",
    fail_log_prefix="PYTHON CO-SIMULATION FAILED",
    panel_color="blue",
)


def run_all_python_tests(
    filter_pattern: Optional[str] = None,
    fail_fast: bool = False,
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False,
    ci_sleep: float = 0.0,
    debug_math: bool = False,
    extra_args: Optional[List[str]] = None
) -> int:
    project_root = get_project_root()
    test_dir = _test_dir(project_root)

    if log_dir is None:
        log_dir = project_root / "out" / "python_reports"
    log_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(project_root=project_root, log_dir=log_dir,
                     fail_fast=fail_fast, verbose=verbose, ci_sleep=ci_sleep,
                     extra={"filter_pattern": filter_pattern,
                            "debug_math": debug_math,
                            "extra_args": extra_args})
    hooks = _CocotbHooks()
    hooks.prepare_all(ctx)

    test_files = discover_python_tests(test_dir, filter_pattern)
    return run_sequential(_SPEC, hooks, [f.name for f in test_files], ctx, dry_run=dry_run)
