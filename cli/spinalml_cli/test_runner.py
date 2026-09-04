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

console = Console(force_terminal=True)

def get_project_root() -> Path:
    return CLI_DIR.parent.resolve()

def setup_tool_env() -> dict:
    """Prepares environment with verilator and compiler toolchains on PATH."""
    env = os.environ.copy()
    
    oss_bin = TOOLS_DIR / "oss-cad-suite" / "bin"
    oss_lib = TOOLS_DIR / "oss-cad-suite" / "lib"
    w64_bin = TOOLS_DIR / "w64devkit" / "bin"
    verilator_root = TOOLS_DIR / "oss-cad-suite" / "share" / "verilator"
    
    new_paths = [str(oss_bin), str(oss_lib)]
    if verilator_root.exists():
        env["VERILATOR_ROOT"] = str(verilator_root)
        verilator_bin = verilator_root / "bin"
        if verilator_bin.exists():
            new_paths.append(str(verilator_bin))
            
    if w64_bin.exists():
        new_paths.append(str(w64_bin))
        
    mill_bin = get_bin_path("mill")
    if mill_bin.exists() and str(mill_bin.parent) not in new_paths:
        new_paths.insert(0, str(mill_bin.parent))
        
    existing_path = env.get("PATH", "") or env.get("Path", "")
    combined_path = os.pathsep.join(new_paths) + os.pathsep + existing_path
    env["PATH"] = combined_path
    env["Path"] = combined_path
    return env

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

def run_all_tests(
    filter_pattern: Optional[str] = None,
    fail_fast: bool = False,
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> int:
    project_root = get_project_root()
    test_src = project_root / "spinalML" / "test" / "src"
    
    if log_dir is None:
        log_dir = project_root / "out" / "test_reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    mill_bin = get_bin_path("mill")
    if not mill_bin.exists():
        console.print(f"[bold red]Error:[/] Mill executable not found at {mill_bin}. Run 'spinalml setup' first.", style="red")
        return 1
        
    tests = discover_tests(test_src, filter_pattern)
    total_tests = len(tests)
    
    if total_tests == 0:
        console.print("[yellow]No tests found matching criteria.[/]")
        return 0
        
    if dry_run:
        console.print(Panel(f"[bold cyan]Discovered {total_tests} test suites[/] (Dry-run mode)"))
        for i, t in enumerate(tests, 1):
            console.print(f"  [dim]{i:2d}.[/] {t}")
        return 0

    console.print(Panel(
        f"[bold green]SpinalML Sequential Test Suite Runner[/]\n"
        f"Discovered [bold]{total_tests}[/] test suites | Logs: [cyan]{log_dir}[/]\n"
        f"Mode: [bold]Sequential (1-by-1)[/] to prevent Verilator/G++ RAM exhaustion on Windows",
        border_style="green"
    ))
    
    env = setup_tool_env()
    
    passed_tests: List[Tuple[str, float]] = []
    failed_tests: List[Tuple[str, float, Path]] = []
    
    total_start_time = time.time()
    
    for idx, test_fqcn in enumerate(tests, 1):
        progress_str = f"[{idx:2d}/{total_tests:2d}]"
        console.print(f"{progress_str} Running [bold blue]{test_fqcn}[/]...")
        sys.stdout.flush()
        
        test_start = time.time()
        cmd = [str(mill_bin), "spinalML.test.testOnly", test_fqcn]
        
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
                passed_tests.append((test_fqcn, duration))
            else:
                log_file = log_dir / f"{test_fqcn}.log"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== TEST RUN FAILED: {test_fqcn} ===\n")
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
                        title=f"[bold red]Failure Output: {test_fqcn}[/]",
                        border_style="red"
                    ))
                    sys.stdout.flush()
                failed_tests.append((test_fqcn, duration, log_file))
                
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
                console.print(Panel(str(e), title=f"[bold red]Exception: {test_fqcn}[/]", border_style="red"))
                sys.stdout.flush()
            failed_tests.append((test_fqcn, duration, log_dir / f"{test_fqcn}_exception.log"))
            if fail_fast:
                sys.stdout.flush()
                break

    total_duration = time.time() - total_start_time
    
    # Render final summary table
    table = Table(title="Test Execution Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    table.add_row("Total Suites Discovered", str(total_tests))
    table.add_row("Suites Executed", str(len(passed_tests) + len(failed_tests)))
    table.add_row("Passed", f"[bold green]{len(passed_tests)}[/]")
    table.add_row("Failed", f"[bold red]{len(failed_tests)}[/]" if failed_tests else "0")
    table.add_row("Total Time", f"{total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    console.print()
    console.print(table)
    
    if failed_tests:
        console.print("\n[bold red]Failed Tests Summary:[/]")
        fail_table = Table(border_style="red")
        fail_table.add_column("Test Class", style="bold red")
        fail_table.add_column("Duration", justify="right")
        fail_table.add_column("Log File", style="dim")
        
        for name, dur, log_p in failed_tests:
            rel_log = log_p.relative_to(project_root) if log_p.is_relative_to(project_root) else log_p
            fail_table.add_row(name, f"{dur:.2f}s", str(rel_log))
        console.print(fail_table)
        return 1
    else:
        console.print(f"\n[bold green]All {len(passed_tests)} tests passed successfully![/]")
        return 0
