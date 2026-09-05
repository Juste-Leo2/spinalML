# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import CLI_DIR, TOOLS_DIR, get_bin_path
from .test_runner import setup_tool_env, get_project_root

console = Console(force_terminal=True)

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

def run_all_formal_tests(
    filter_pattern: Optional[str] = None,
    fail_fast: bool = False,
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
    timeout: int = 900,
    verbose: bool = False
) -> int:
    project_root = get_project_root()
    formal_src = project_root / "spinalML" / "test" / "src" / "spinalML" / "symbolicTest"
    
    if log_dir is None:
        log_dir = project_root / "out" / "formal_reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    mill_bin = get_bin_path("mill")
    if not mill_bin.exists():
        console.print(f"[bold red]Error:[/] Mill executable not found at {mill_bin}. Run 'spinalml setup' first.", style="red")
        return 1
        
    specs = discover_formal_specs(formal_src, filter_pattern)
    total_specs = len(specs)
    
    if total_specs == 0:
        console.print("[yellow]No formal verification suites found matching criteria.[/]")
        return 0
        
    if dry_run:
        console.print(Panel(f"[bold cyan]Discovered {total_specs} Formal Verification Suites[/] (Dry-run mode)"))
        for i, s in enumerate(specs, 1):
            console.print(f"  [dim]{i:2d}.[/] {s}")
        return 0

    console.print(Panel(
        f"[bold cyan]SpinalML Formal Verification Runner (SymbiYosys / BMC)[/]\n"
        f"Discovered [bold]{total_specs}[/] formal suites | Logs: [cyan]{log_dir}[/]\n"
        f"Mode: [bold]Sequential (1-by-1)[/] | Timeout per suite: [bold]{timeout}s[/]",
        border_style="cyan"
    ))
    
    env = setup_tool_env()
    
    passed_specs: List[Tuple[str, float]] = []
    failed_specs: List[Tuple[str, float, Path]] = []
    
    total_start_time = time.time()
    
    for idx, spec_fqcn in enumerate(specs, 1):
        progress_str = f"[{idx:2d}/{total_specs:2d}]"
        console.print(f"{progress_str} Verifying [bold magenta]{spec_fqcn}[/]...")
        sys.stdout.flush()
        
        spec_start = time.time()
        cmd = [str(mill_bin), "--no-server", "--disable-ticker", "spinalML.test.runMain", spec_fqcn]
        
        try:
            res = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - spec_start
            
            # Check for failure indications in stdout/stderr even if exit code is 0
            is_success = (res.returncode == 0) and ("Assert failed" not in res.stdout) and ("FAIL" not in res.stdout or "SUCCESS" in res.stdout)
            
            if is_success:
                console.print(f"       -> [bold green]PASS[/] ({duration:5.2f}s)")
                sys.stdout.flush()
                passed_specs.append((spec_fqcn, duration))
            else:
                log_file = log_dir / f"{spec_fqcn}.log"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== FORMAL VERIFICATION FAILED: {spec_fqcn} ===\n")
                    f.write(f"Duration: {duration:.2f}s\n")
                    f.write(f"Command: {' '.join(cmd)}\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(res.stdout)
                    f.write("\n=== STDERR ===\n")
                    f.write(res.stderr)
                    
                console.print(f"       -> [bold red]FAIL[/] ({duration:5.2f}s) -> [dim]{log_file.relative_to(project_root)}[/]")
                combined_output = (res.stderr or "") + (res.stdout or "")
                if "checksum format error" in combined_output or "scalaCompilerBridge" in combined_output:
                    console.print("       [bold yellow]↳ Detected Coursier cache checksum corruption.[/bold yellow] Run [bold cyan]python cli/main.py clean-cache[/bold cyan] to clear.")
                sys.stdout.flush()
                if verbose:
                    out_trace = res.stderr.strip() or res.stdout.strip() or "No output captured."
                    console.print(Panel(
                        out_trace,
                        title=f"[bold red]Formal Failure Output: {spec_fqcn}[/]",
                        border_style="red"
                    ))
                    sys.stdout.flush()
                failed_specs.append((spec_fqcn, duration, log_file))
                
                if fail_fast:
                    console.print("\n[bold red]Stopping early due to --fail-fast.[/]")
                    sys.stdout.flush()
                    break
        except subprocess.TimeoutExpired:
            duration = time.time() - spec_start
            log_file = log_dir / f"{spec_fqcn}_timeout.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"=== FORMAL VERIFICATION TIMED OUT ({timeout}s): {spec_fqcn} ===\n")
            console.print(f"       -> [bold red]TIMEOUT[/] (>{timeout}s)")
            sys.stdout.flush()
            if verbose:
                console.print(Panel(f"Timed out after {timeout} seconds", title=f"[bold red]Timeout: {spec_fqcn}[/]", border_style="red"))
                sys.stdout.flush()
            failed_specs.append((spec_fqcn, duration, log_file))
            if fail_fast:
                sys.stdout.flush()
                break
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Aborted by user.[/]")
            sys.stdout.flush()
            return 130
        except Exception as e:
            duration = time.time() - spec_start
            console.print(f"       -> [bold red]ERROR[/] ({duration:5.2f}s): {e}")
            sys.stdout.flush()
            if verbose:
                console.print(Panel(str(e), title=f"[bold red]Exception: {spec_fqcn}[/]", border_style="red"))
                sys.stdout.flush()
            failed_specs.append((spec_fqcn, duration, log_dir / f"{spec_fqcn}_exception.log"))
            if fail_fast:
                sys.stdout.flush()
                break

    total_duration = time.time() - total_start_time
    
    # Render final summary table
    table = Table(title="Formal Verification Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    table.add_row("Total Formal Suites Discovered", str(total_specs))
    table.add_row("Suites Executed", str(len(passed_specs) + len(failed_specs)))
    table.add_row("Passed", f"[bold green]{len(passed_specs)}[/]")
    table.add_row("Failed", f"[bold red]{len(failed_specs)}[/]" if failed_specs else "0")
    table.add_row("Total Time", f"{total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    console.print()
    console.print(table)
    
    if failed_specs:
        console.print("\n[bold red]Failed Formal Specs Summary:[/]")
        fail_table = Table(border_style="red")
        fail_table.add_column("Formal Spec", style="bold red")
        fail_table.add_column("Duration", justify="right")
        fail_table.add_column("Log File", style="dim")
        
        for name, dur, log_p in failed_specs:
            rel_log = log_p.relative_to(project_root) if log_p.is_relative_to(project_root) else log_p
            fail_table.add_row(name, f"{dur:.2f}s", str(rel_log))
        console.print(fail_table)
        return 1
    else:
        console.print(f"\n[bold green]All {len(passed_specs)} formal verification suites passed successfully![/]")
        return 0
