# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Shared sequential-runner engine (test-all, test-all-formal, test-all-python).

The three runners were line-for-line the same state machine
(discover -> dry-run -> sequential loop with try/except -> logs -> tables);
this engine holds the loop once. Per-runner differences plug in via
RunnerSpec (display strings) and RunnerHooks (commands, classification,
hints). Behavior is intentionally identical to the pre-refactor runners,
down to quirks (e.g. scala/formal exception logs are referenced in the fail
table but never written to disk).
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(force_terminal=True)

PASS, SKIP, FAIL = "pass", "skip", "fail"
ABORT, TIMEOUT, ERROR = "abort", "timeout", "error"

RETRY_DELAY_S = 2.0


class AbortItem(Exception):
    """Raised by hooks when an item cannot run: engine prints the message, returns 1."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class RunnerSpec:
    """All display strings of a runner (moved verbatim from the originals)."""

    title: str            # header panel, line 1 (rich markup)
    mode_line: str        # header panel, mode line (rich markup, no trailing newline)
    item_noun: str        # "test suites" | "formal suites" | "test files"
    run_verb: str         # "Running" | "Verifying"
    run_verb_color: str   # "blue" | "magenta"
    total_label: str      # summary metric, e.g. "Total Suites Discovered"
    exec_label: str       # summary metric, e.g. "Suites Executed"
    summary_title: str    # summary table title
    fail_summary_title: str   # red section title, e.g. "Failed Tests Summary:"
    fail_first_col: str   # fail table first column, e.g. "Test Class"
    all_pass_msg: str     # final green line, e.g. "All {n} tests passed successfully!"
    empty_msg: str        # yellow line when nothing discovered
    dry_title: str        # dry-run panel title, e.g. "Discovered {n} test suites"
    fail_log_prefix: str  # log header, e.g. "TEST RUN FAILED"
    panel_color: str      # header/dry-run panel border ("green" | "cyan" | "blue")


@dataclass
class RunContext:
    """Per-run state handed to hooks."""

    project_root: Path
    log_dir: Path
    fail_fast: bool
    verbose: bool
    ci_sleep: float
    extra: Dict[str, Any]
    max_retries: int = 0


class RunnerHooks:
    """Per-runner behavior. Defaults cover the scala runner; formal/cocotb override."""

    def prepare_all(self, ctx: RunContext) -> None:
        """Early side effects before the loop (cocotb: orphan cleanup, math log)."""

    def check_tool(self, ctx: RunContext) -> Optional[str]:
        """Returns the full error line if the run cannot start, else None."""
        return None

    def tool_error_style(self) -> Optional[str]:
        """console.print style for the check_tool error line (scala/formal: red)."""
        return None

    def build_env(self, ctx: RunContext) -> dict:
        from .env import setup_tool_env
        return setup_tool_env()

    def build_cmd(self, item: str, ctx: RunContext) -> List[str]:
        """Returns the subprocess argv. May raise AbortItem to stop the run with code 1."""
        raise NotImplementedError

    def run_timeout(self, item: str, ctx: RunContext) -> Optional[float]:
        return None

    def classify(self, res: subprocess.CompletedProcess, item: str, ctx: RunContext) -> str:
        return PASS if res.returncode == 0 else FAIL

    def skip_suffix(self, item: str, ctx: RunContext) -> str:
        return ""

    def log_name(self, item: str, kind: str, ctx: RunContext) -> str:
        suffix = {"fail": ".log", "timeout": "_timeout.log", "exception": "_exception.log"}[kind]
        return f"{item}{suffix}"

    def log_header(self, item: str, kind: str, ctx: RunContext) -> str:
        """Log file header line (without trailing newline). Overridden per runner."""
        return f"=== {item} ({kind}) ==="

    def exception_log_content(self, item: str, exc: Exception, cmd: List[str],
                              ctx: RunContext) -> Optional[str]:
        """Content of the exception log file, or None to reference-but-not-write (scala/formal)."""
        return None

    def failure_hint(self, output: str) -> Optional[str]:
        return None

    def detail_title(self, item: str, kind: str, ctx: RunContext) -> str:
        titles = {"fail": f"Failure Output: {item}",
                  "timeout": f"Timeout: {item}",
                  "exception": f"Exception: {item}"}
        return titles[kind]

    def timeout_line(self, item: str, timeout: float, duration: float) -> str:
        return f"       -> [bold red]TIMEOUT[/] (>{timeout}s)"

    def timeout_panel_body(self, item: str, timeout: float) -> str:
        return f"Timed out after {timeout} seconds"


def _print_header(spec: RunnerSpec, total: int, log_dir: Path) -> None:
    console.print(Panel(
        f"{spec.title}\n"
        f"Discovered [bold]{total}[/] {spec.item_noun} | Logs: [cyan]{log_dir}[/]\n"
        f"{spec.mode_line}",
        border_style=spec.panel_color
    ))


def _write_fail_log(log_file: Path, header: str, duration: float,
                    cmd: List[str], res: subprocess.CompletedProcess) -> None:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(res.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(res.stderr)


def _print_verbose_panel(title: str, body: str) -> None:
    console.print(Panel(body, title=f"[bold red]{title}[/]", border_style="red"))
    sys.stdout.flush()


def _print_summary(spec: RunnerSpec, total: int, passed: list, failed: list,
                   total_duration: float, project_root: Path, flaky: Optional[list] = None) -> int:
    table = Table(title=spec.summary_title, border_style=spec.panel_color)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row(spec.total_label, str(total))
    table.add_row(spec.exec_label, str(len(passed) + len(failed)))
    table.add_row("Passed", f"[bold green]{len(passed)}[/]")
    table.add_row("Failed", f"[bold red]{len(failed)}[/]" if failed else "0")
    if flaky:
        table.add_row("Flaky (passed on retry)", f"[bold yellow]{len(flaky)}[/]")
    table.add_row("Total Time", f"{total_duration:.1f}s ({total_duration / 60:.1f} min)")

    console.print()
    console.print(table)

    if failed:
        console.print(f"\n[bold red]{spec.fail_summary_title}[/]")
        fail_table = Table(border_style="red")
        fail_table.add_column(spec.fail_first_col, style="bold red")
        fail_table.add_column("Duration", justify="right")
        fail_table.add_column("Log File", style="dim")

        for name, dur, log_p in failed:
            rel_log = log_p.relative_to(project_root) if log_p.is_relative_to(project_root) else log_p
            fail_table.add_row(name, f"{dur:.2f}s", str(rel_log))
        console.print(fail_table)
        return 1
    console.print(f"\n[bold green]{spec.all_pass_msg.format(n=len(passed))}[/]")
    return 0


def check_tool_or_report(hooks: RunnerHooks, ctx: RunContext) -> bool:
    """Runs the tool preflight (before discovery, like the originals). True to proceed."""
    err = hooks.check_tool(ctx)
    if err is None:
        return True
    console.print(err, style=hooks.tool_error_style())
    return False


def _run_attempt(spec: RunnerSpec, hooks: RunnerHooks, item: str, ctx: RunContext,
                 cmd: List[str], env: dict, timeout: Optional[float],
                 attempt: int, attempts: int) -> Tuple[str, float, Optional[Path]]:
    """Runs ONE attempt: subprocess + classify + log + print.

    Returns (outcome, duration, fail_log_or_None). Only FAIL is retried by the
    caller; TIMEOUT/ERROR are final, ABORT returns 130 immediately.
    """
    def attempt_log(kind: str) -> Path:
        base = hooks.log_name(item, kind, ctx)
        if attempts > 1:
            stem, dot, ext = base.rpartition(".")
            base = f"{stem}.attempt{attempt}.{ext}" if dot else f"{base}.attempt{attempt}"
        return ctx.log_dir / base

    start = time.time()
    try:
        res = subprocess.run(
            cmd,
            cwd=str(ctx.project_root),
            env=env,
            capture_output=True,
            text=True,
            **({"timeout": timeout} if timeout is not None else {})
        )
        duration = time.time() - start
        outcome = hooks.classify(res, item, ctx)

        if outcome == PASS:
            suffix = f", attempt {attempt}/{attempts}" if (attempts > 1 and attempt > 1) else ""
            console.print(f"       -> [bold green]PASS[/] ({duration:5.2f}s{suffix})")
            sys.stdout.flush()
            return PASS, duration, None
        if outcome == SKIP:
            console.print(f"       -> [dim yellow]SKIP[/] ({duration:5.2f}s){hooks.skip_suffix(item, ctx)}")
            sys.stdout.flush()
            return SKIP, duration, None

        log_file = attempt_log("fail")
        _write_fail_log(log_file, hooks.log_header(item, "fail", ctx),
                        duration, cmd, res)
        console.print(f"       -> [bold red]FAIL[/] ({duration:5.2f}s) -> [dim]{log_file.relative_to(ctx.project_root)}[/]")
        hint = hooks.failure_hint((res.stderr or "") + (res.stdout or ""))
        if hint:
            console.print(hint)
        sys.stdout.flush()
        if ctx.verbose:
            _print_verbose_panel(hooks.detail_title(item, "fail", ctx),
                                 res.stderr.strip() or res.stdout.strip() or "No output captured.")
        return FAIL, duration, log_file
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        log_file = attempt_log("timeout")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{hooks.log_header(item, 'timeout', ctx)}\n")
        console.print(hooks.timeout_line(item, timeout, duration))
        sys.stdout.flush()
        if ctx.verbose:
            _print_verbose_panel(hooks.detail_title(item, "timeout", ctx),
                                 hooks.timeout_panel_body(item, timeout))
        return TIMEOUT, duration, log_file
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Aborted by user.[/]")
        sys.stdout.flush()
        return ABORT, 0.0, None
    except Exception as e:
        duration = time.time() - start
        console.print(f"       -> [bold red]ERROR[/] ({duration:5.2f}s): {e}")
        sys.stdout.flush()
        if ctx.verbose:
            _print_verbose_panel(hooks.detail_title(item, "exception", ctx), str(e))
        content = hooks.exception_log_content(item, e, cmd, ctx)
        log_file = attempt_log("exception")
        if content is not None:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(content)
        return ERROR, duration, log_file


def run_sequential(spec: RunnerSpec, hooks: RunnerHooks, items: List[str],
                   ctx: RunContext, dry_run: bool = False) -> int:
    """Executes the shared sequential loop. Returns 0 (ok), 1 (failures) or 130 (abort)."""
    total = len(items)

    if total == 0:
        console.print(f"[yellow]{spec.empty_msg}[/]")
        return 0

    if dry_run:
        console.print(Panel(f"[bold {spec.panel_color}]{spec.dry_title.format(n=total)}[/] (Dry-run mode)",
                            border_style=spec.panel_color))
        for i, item in enumerate(items, 1):
            console.print(f"  [dim]{i:2d}.[/] {item}")
        return 0

    _print_header(spec, total, ctx.log_dir)

    try:
        env = hooks.build_env(ctx)
    except AbortItem as e:
        console.print(f"[bold red]Error:[/] {e.message}")
        return 1

    passed: List[Tuple[str, float]] = []
    failed: List[Tuple[str, float, Path]] = []
    flaky: List[str] = []

    total_start_time = time.time()

    for idx, item in enumerate(items, 1):
        progress_str = f"[{idx:2d}/{total:2d}]"
        console.print(f"{progress_str} {spec.run_verb} [bold {spec.run_verb_color}]{item}[/]...")
        sys.stdout.flush()

        try:
            cmd = hooks.build_cmd(item, ctx)
        except AbortItem as e:
            console.print(f"[bold red]Error:[/] {e.message}")
            return 1
        timeout = hooks.run_timeout(item, ctx)

        attempts = 1 + max(0, ctx.max_retries)
        outcome: Optional[str] = None
        for attempt in range(1, attempts + 1):
            outcome, duration, log_file = _run_attempt(
                spec, hooks, item, ctx, cmd, env, timeout, attempt, attempts)
            if outcome == ABORT:
                return 130
            if outcome == PASS:
                passed.append((item, duration))
                if attempt > 1:
                    flaky.append(item)
                break
            if outcome == SKIP:
                break
            if outcome in (TIMEOUT, ERROR):
                # TIMEOUT/ERROR never retry (FAIL-only policy).
                assert log_file is not None
                failed.append((item, duration, log_file))
                break
            # FAIL: retry while attempts remain.
            assert log_file is not None
            if attempt == attempts:
                failed.append((item, duration, log_file))
                break
            console.print(f"       [yellow]Retrying ({attempt + 1}/{attempts}) in {RETRY_DELAY_S:.0f}s...[/]")
            sys.stdout.flush()
            time.sleep(RETRY_DELAY_S)

        if outcome in (FAIL, TIMEOUT, ERROR) and ctx.fail_fast:
            console.print("\n[bold red]Stopping early due to --fail-fast.[/]")
            sys.stdout.flush()
            break

        if ctx.ci_sleep > 0 and idx < total:
            time.sleep(ctx.ci_sleep)

    total_duration = time.time() - total_start_time
    return _print_summary(spec, total, passed, failed, total_duration, ctx.project_root, flaky)
