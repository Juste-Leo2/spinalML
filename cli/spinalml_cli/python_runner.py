# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import CLI_DIR, TOOLS_DIR, get_project_root
from .test_runner import setup_tool_env

console = Console(force_terminal=True)

def find_python_interpreter() -> Optional[str]:
    """Finds an external Python interpreter capable of running pytest/cocotb."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    root = get_project_root()
    # Check local .venv
    venv_win = root / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        return str(venv_win)
    venv_unix = root / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return str(venv_unix)
    # Check system / PATH
    for candidate in ["python3.12", "python3", "python"]:
        found = shutil.which(candidate)
        if found:
            return found
    return None


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
    test_dir = project_root / "tests" / "python"

    if log_dir is None:
        log_dir = project_root / "out" / "python_reports"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Prevent VPATH ../verilator.o pollution: ensure no orphan object files exist at sim_build root
    sim_build_root = project_root / "sim_build"
    if sim_build_root.exists():
        for orphan in sim_build_root.glob("*.o"):
            try:
                orphan.unlink()
            except OSError:
                pass

    test_files = discover_python_tests(test_dir, filter_pattern)
    total_files = len(test_files)

    if total_files == 0:
        console.print("[yellow]No Python co-simulation tests found matching criteria.[/]")
        return 0

    if dry_run:
        console.print(Panel(f"[bold blue]Discovered {total_files} Python Co-Simulation Test Files[/] (Dry-run mode)", border_style="blue"))
        for i, f in enumerate(test_files, 1):
            console.print(f"  [dim]{i:2d}.[/] {f.name}")
        return 0

    console.print(Panel(
        f"[bold blue]SpinalML Python Co-Simulation Suite Runner (Pytest / Cocotb)[/]\n"
        f"Discovered [bold]{total_files}[/] test files | Logs: [cyan]{log_dir}[/]\n"
        f"Mode: [bold]Sequential (file-by-file)[/] to prevent RAM/VPI exhaustion on self-hosted runners",
        border_style="blue"
    ))

    env = setup_tool_env()
    cli_str = str(CLI_DIR)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cli_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else cli_str
    if ci_sleep > 0:
        env["SM_CI_SLEEP"] = str(ci_sleep)

    passed_tests: List[Tuple[str, float]] = []
    failed_tests: List[Tuple[str, float, Path]] = []

    total_start_time = time.time()

    for idx, test_file in enumerate(test_files, 1):
        progress_str = f"[{idx:2d}/{total_files:2d}]"
        console.print(f"{progress_str} Running [bold blue]{test_file.name}[/]...")
        sys.stdout.flush()

        test_start = time.time()
        rel_test_path = str(test_file.relative_to(project_root))
        py_bin = find_python_interpreter()
        if not py_bin:
            console.print("[bold red]Error:[/] Could not locate a Python interpreter for pytest/cocotb.\n"
                          "Python co-simulations require a local environment with pytest & cocotb.\n"
                          "Install requirements with: pip install -r requirements.txt")
            return 1
        cmd = [py_bin, "-m", "pytest", rel_test_path]


        # Only pass -k to pytest if the filter was targeting a specific test inside the file (content match, not filename)
        matched_filename = bool(re.search(filter_pattern, test_file.stem, re.IGNORECASE) or re.search(filter_pattern, test_file.name, re.IGNORECASE)) if filter_pattern else False
        if filter_pattern and not matched_filename:
            cmd.extend(["-k", filter_pattern])

        if fail_fast:
            cmd.append("-x")
        if debug_math:
            cmd.append("--debug-math")
        if extra_args:
            cmd.extend(extra_args)

        # Run with -s to keep output captured properly
        cmd.append("-s")

        try:
            res = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True
            )
            duration = time.time() - test_start

            if res.returncode == 0:
                console.print(f"       -> [bold green]PASS[/] ({duration:5.2f}s)")
                sys.stdout.flush()
                passed_tests.append((test_file.name, duration))
            elif res.returncode == 5 and filter_pattern:
                # Pytest exit code 5: no tests collected matching pattern inside this file
                console.print(f"       -> [dim yellow]SKIP[/] ({duration:5.2f}s) (no tests matched '{filter_pattern}')")
                sys.stdout.flush()
            else:
                log_file = log_dir / f"{test_file.stem}.log"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== PYTHON CO-SIMULATION FAILED: {test_file.name} ===\n")
                    f.write(f"Duration: {duration:.2f}s\n")
                    f.write(f"Command: {' '.join(cmd)}\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(res.stdout)
                    f.write("\n=== STDERR ===\n")
                    f.write(res.stderr)

                console.print(f"       -> [bold red]FAIL[/] ({duration:5.2f}s) -> [dim]{log_file.relative_to(project_root)}[/]")
                sys.stdout.flush()

                if verbose:
                    out_trace = res.stderr.strip() or res.stdout.strip() or "No output captured."
                    console.print(Panel(
                        out_trace,
                        title=f"[bold red]Failure Output: {test_file.name}[/]",
                        border_style="red"
                    ))
                    sys.stdout.flush()

                failed_tests.append((test_file.name, duration, log_file))

                if fail_fast:
                    console.print("\n[bold red]Stopping early due to --fail-fast.[/]")
                    sys.stdout.flush()
                    break

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Aborted by user.[/]")
            sys.stdout.flush()
            return 130
        except Exception as e:
            duration = time.time() - test_start
            console.print(f"       -> [bold red]ERROR[/] ({duration:5.2f}s): {e}")
            sys.stdout.flush()
            if verbose:
                console.print(Panel(str(e), title=f"[bold red]Exception: {test_file.name}[/]", border_style="red"))
                sys.stdout.flush()
            exc_log_file = log_dir / f"{test_file.stem}_exception.log"
            with open(exc_log_file, "w", encoding="utf-8") as f:
                f.write(f"=== EXCEPTION RUNNING {test_file.name} ===\n{str(e)}\n")
            failed_tests.append((test_file.name, duration, exc_log_file))
            if fail_fast:
                sys.stdout.flush()
                break

        if ci_sleep > 0 and idx < total_files:
            time.sleep(ci_sleep)

    total_duration = time.time() - total_start_time

    # Render final summary table
    table = Table(title="Python Co-Simulation Summary", border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total Files Discovered", str(total_files))
    table.add_row("Files Executed", str(len(passed_tests) + len(failed_tests)))
    table.add_row("Passed", f"[bold green]{len(passed_tests)}[/]")
    table.add_row("Failed", f"[bold red]{len(failed_tests)}[/]" if failed_tests else "0")
    table.add_row("Total Time", f"{total_duration:.1f}s ({total_duration/60:.1f} min)")

    console.print()
    console.print(table)

    if failed_tests:
        console.print("\n[bold red]Failed Tests Summary:[/]")
        fail_table = Table(border_style="red")
        fail_table.add_column("Test File", style="bold red")
        fail_table.add_column("Duration", justify="right")
        fail_table.add_column("Log File", style="dim")

        for name, dur, log_p in failed_tests:
            rel_log = log_p.relative_to(project_root) if log_p.is_relative_to(project_root) else log_p
            fail_table.add_row(name, f"{dur:.2f}s", str(rel_log))
        console.print(fail_table)
        return 1
    else:
        console.print(f"\n[bold green]All {len(passed_tests)} Python test files passed successfully![/]")
        return 0
