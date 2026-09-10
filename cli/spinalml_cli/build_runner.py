# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import re
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from .config import CLI_DIR, TOOLS_DIR, get_bin_path
from .board import load_board_config, resolve_constraints_file, parse_frequency

console = Console(legacy_windows=False)

def get_eda_env() -> Dict[str, str]:
    """Sets up the environment variables for OSS CAD Suite tools on Windows and Linux."""
    env = os.environ.copy()
    oss_bin = TOOLS_DIR / "oss-cad-suite" / "bin"
    oss_lib = TOOLS_DIR / "oss-cad-suite" / "lib"
    w64_bin = TOOLS_DIR / "w64devkit" / "bin"

    new_paths = []
    if oss_bin.exists():
        new_paths.append(str(oss_bin))
    if oss_lib.exists():
        new_paths.append(str(oss_lib))
    if w64_bin.exists():
        new_paths.append(str(w64_bin))

    existing_path = env.get("PATH", "") or env.get("Path", "")
    combined_path = os.pathsep.join(new_paths) + os.pathsep + existing_path
    env["PATH"] = combined_path
    env["Path"] = combined_path
    return env

def detect_top_module(v_files: List[Path], preferred_top: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Scans Verilog files to detect the top-level module and its port names.
    Prioritizes:
    1. preferred_top (if explicitly passed)
    2. UartSoC
    3. top
    4. First module declaration found
    """
    module_regex = re.compile(r'^\s*module\s+(\w+)\s*(?:#\s*\([^)]*\)\s*)?\s*\((.*?)\);', re.MULTILINE | re.DOTALL)
    modules: Dict[str, List[str]] = {}

    for vf in v_files:
        try:
            content = vf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for match in module_regex.finditer(content):
            mod_name = match.group(1)
            port_block = match.group(2)
            ports = []
            for line in port_block.split(','):
                line = re.sub(r'//.*', '', line).strip()
                m_port = re.search(r'(?:input|output|inout)\s+(?:(?:wire|reg)\s+)?(?:\[[^\]]+\]\s*)?(\w+)', line)
                if m_port:
                    ports.append(m_port.group(1))
            modules[mod_name] = ports

    if preferred_top and preferred_top in modules:
        return preferred_top, modules[preferred_top]

    if "UartSoC" in modules:
        return "UartSoC", modules["UartSoC"]
    if "top" in modules:
        return "top", modules["top"]

    if modules:
        first_mod = list(modules.keys())[0]
        return first_mod, modules[first_mod]

    return preferred_top or "top", []

def filter_unique_verilog_files(v_files: List[Path], top_module: str) -> List[Path]:
    """
    Selects Verilog files to pass to Yosys, avoiding duplicate module definitions across files.
    Prioritizes the file containing the top-level module (e.g. UartSoC.v or top.v),
    and only includes additional files if they define new modules not already defined.
    """
    if len(v_files) <= 1:
        return v_files

    module_regex = re.compile(r'^\s*module\s+(\w+)', re.MULTILINE)
    file_modules: Dict[Path, List[str]] = {}

    for vf in v_files:
        try:
            content = vf.read_text(encoding="utf-8", errors="ignore")
            mods = module_regex.findall(content)
            file_modules[vf] = mods
        except Exception:
            file_modules[vf] = []

    def sort_key(vf: Path) -> int:
        mods = file_modules.get(vf, [])
        if top_module in mods:
            return 0
        if vf.stem == top_module:
            return 0
        return 1

    sorted_files = sorted(v_files, key=sort_key)
    selected: List[Path] = []
    defined_modules = set()

    for vf in sorted_files:
        mods = file_modules.get(vf, [])
        new_mods = set(mods) - defined_modules
        if new_mods or not defined_modules:
            selected.append(vf)
            defined_modules.update(mods)

    return selected if selected else v_files

def extract_yosys_resources(synth_json_path: Path, top_module: str) -> Dict[str, int]:
    """
    Extracts cell utilization from Yosys JSON netlist (synth.json).
    Accurately sums LUTs, FFs, BRAMs, and DSPs across all modules or top module.
    """
    res = {"LUT4": 0, "FF": 0, "BRAM": 0, "DSP": 0}
    if not synth_json_path.exists():
        return res

    try:
        with open(synth_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        modules = data.get("modules", {})
        top_data = modules.get(top_module) or modules.get(f"\\{top_module}") or (list(modules.values())[0] if modules else {})
        cells = top_data.get("cells", {})

        for c_name, c_info in cells.items():
            ctype = c_info.get("type", "")
            # LUTs
            if ctype in ("LUT1", "LUT2", "LUT3", "LUT4", "SB_LUT4") or ctype.startswith("LUT"):
                res["LUT4"] += 1
            # FFs (DFF, DFFE, DFFR, DFFRE, DFFS, DFFSE, etc.)
            elif ctype.startswith("DFF") or ctype.startswith("FD") or ctype.startswith("SB_DFF"):
                res["FF"] += 1
            # BRAMs (DPB, SP, SDPB, BSRAM, etc.)
            elif ctype in ("DPB", "SP", "SDPB", "BSRAM", "RAM16SDP4") or "RAM" in ctype:
                if ctype in ("DPB", "SP", "SDPB", "BSRAM"):
                    res["BRAM"] += 1
            # DSPs (MULT18X18, MULT9X9, DSP, etc.)
            elif "MULT" in ctype or "DSP" in ctype:
                if ctype == "MULT18X18":
                    res["DSP"] += 1
                elif ctype == "MULT9X9":
                    res["DSP"] += 0.5
                else:
                    res["DSP"] += 1

        res["DSP"] = int(res["DSP"])
    except Exception:
        pass

    return res

def extract_pnr_resources(pnr_report_path: Path) -> Tuple[Dict[str, int], Optional[float]]:
    """
    Extracts cell utilization and achieved Fmax from nextpnr JSON report (pnr_report.json).
    Uses exact key matches to avoid partial string matching (e.g. MUX2_LUT8 overwriting LUT4).
    """
    res = {}
    fmax = None
    if not pnr_report_path.exists():
        return res, fmax

    try:
        with open(pnr_report_path, "r", encoding="utf-8") as rf:
            rep_data = json.load(rf)
            util = rep_data.get("utilization", {})

            if "LUT4" in util:
                res["LUT4"] = util["LUT4"].get("used", 0)
            elif "LUT" in util:
                res["LUT4"] = util["LUT"].get("used", 0)

            if "DFF" in util:
                res["FF"] = util["DFF"].get("used", 0)
            elif "FF" in util:
                res["FF"] = util["FF"].get("used", 0)

            if "BSRAM" in util:
                res["BRAM"] = util["BSRAM"].get("used", 0)
            elif "BRAM" in util:
                res["BRAM"] = util["BRAM"].get("used", 0)

            dsp_total = 0.0
            if "MULT18X18" in util:
                dsp_total += util["MULT18X18"].get("used", 0)
            if "MULT9X9" in util:
                dsp_total += util["MULT9X9"].get("used", 0) * 0.5
            if "DSP" in util:
                dsp_total += util["DSP"].get("used", 0)
            res["DSP"] = int(round(dsp_total))

            fmax_dict = rep_data.get("fmax", {})
            for clk_name, clk_data in fmax_dict.items():
                if isinstance(clk_data, dict) and "achieved" in clk_data:
                    fmax = float(clk_data["achieved"])
                    break
    except Exception:
        pass

    return res, fmax

def ensure_apycula_patched():
    """
    Auto-patches an upstream bug in OSS CAD Suite's apycula/gowin_pack.py where B multiplier
    attribute indices were missing an offset of 2, causing KeyError on IRBY_IREG0BL_0.
    """
    for py_ver in ["python3.11", "python3.10", "python3.12"]:
        apycula_file = TOOLS_DIR / "oss-cad-suite" / "lib" / py_ver / "site-packages" / "apycula" / "gowin_pack.py"
        if apycula_file.exists():
            try:
                content = apycula_file.read_text(encoding="utf-8")
                old_target = """            else:
                if is_even:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}H_{pair_idx * 4 + 1}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}H_{pair_idx * 4 + 1}', "ENABLE"))
                else:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}L_{pair_idx * 4}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}L_{pair_idx * 4}', "ENABLE"))"""
                new_replacement = """            else:
                r_offset = 0 if r == 'A' else 2
                if is_even:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}H_{pair_idx * 4 + 1 + r_offset}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}H_{pair_idx * 4 + 1 + r_offset}', "ENABLE"))
                else:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}L_{pair_idx * 4 + r_offset}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}L_{pair_idx * 4 + r_offset}', "ENABLE"))"""
                if old_target in content:
                    content = content.replace(old_target, new_replacement, 1)
                    apycula_file.write_text(content, encoding="utf-8")
            except Exception:
                pass

def adapt_constraints_for_ports(original_cst: Optional[Path], ports: List[str], target_cst: Path):
    """
    Creates an adapted Gowin CST file mapping the board's pins to the actual top module port names.
    Handles variations like clk, reset_n vs io_resetN, uart_rx vs io_uartRx, uart_tx vs io_uartTx.
    """
    if not original_cst or not original_cst.exists():
        return

    cst_text = original_cst.read_text(encoding="utf-8", errors="ignore")
    lines = cst_text.splitlines()

    # Detect port naming convention used by top module
    has_io_ports = any(p.startswith("io_") for p in ports)

    # Pin dictionary: physical pin -> signal role
    # Tang Primer 20K standard pins:
    # clk -> H11, reset_n/io_resetN -> T10, uart_rx/io_uartRx -> T13, uart_tx/io_uartTx -> M11
    adapted_lines = [
        "// Auto-adapted physical constraints for top module ports",
        "// Generated by SpinalML build engine"
    ]

    # Map signals
    clk_port = "clk" if "clk" in ports else (ports[0] if ports else "clk")
    reset_port = "io_resetN" if "io_resetN" in ports else ("reset_n" if "reset_n" in ports else None)
    rx_port = "io_uartRx" if "io_uartRx" in ports else ("uart_rx" if "uart_rx" in ports else None)
    tx_port = "io_uartTx" if "io_uartTx" in ports else ("uart_tx" if "uart_tx" in ports else None)

    # Clock
    adapted_lines.append(f'IO_LOC "{clk_port}" H11;')
    adapted_lines.append(f'IO_PORT "{clk_port}" IO_TYPE=LVCMOS33 PULL_MODE=UP;')

    # Reset
    if reset_port:
        adapted_lines.append(f'IO_LOC "{reset_port}" T10;')
        adapted_lines.append(f'IO_PORT "{reset_port}" IO_TYPE=LVCMOS33 PULL_MODE=UP;')

    # UART RX
    if rx_port:
        adapted_lines.append(f'IO_LOC "{rx_port}" T13;')
        adapted_lines.append(f'IO_PORT "{rx_port}" IO_TYPE=LVCMOS33 PULL_MODE=UP;')

    # UART TX
    if tx_port:
        adapted_lines.append(f'IO_LOC "{tx_port}" M11;')
        adapted_lines.append(f'IO_PORT "{tx_port}" IO_TYPE=LVCMOS33 PULL_MODE=UP;')

    target_cst.write_text("\n".join(adapted_lines) + "\n", encoding="utf-8")

def format_progress_bar(used: int, capacity: int, width: int = 10) -> str:
    """Generates an ASCII/Unicode progress bar with usage percentage."""
    if capacity <= 0:
        return f"{used:,}"
    pct = (used / capacity) * 100
    filled = int(round((used / capacity) * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"{pct:5.1f} % [{bar}]"

def render_final_report(
    board_cfg: Dict[str, Any],
    top_module: str,
    resources: Dict[str, int],
    timing_info: Dict[str, Any],
    bitstream_path: Optional[Path],
    duration: float,
    stopped_at: str
):
    """Renders the comprehensive hardware resource utilization & timing Rich table."""
    limits = board_cfg.get("limits", {})
    lut_cap = limits.get("lut", 0)
    ff_cap = limits.get("ff", 0)
    bram_cap = limits.get("bram", 0)
    dsp_cap = limits.get("dsp", 0)

    lut_used = resources.get("LUT4", resources.get("LUT", 0))
    ff_used = resources.get("FF", resources.get("DFF", 0))
    bram_used = resources.get("BRAM", resources.get("BSRAM", 0))
    dsp_used = resources.get("DSP", resources.get("MULT", 0))

    title = f"Hardware Resource Utilization ({board_cfg['name']})"
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Resource", style="cyan", width=18)
    table.add_column("Used", justify="right", width=12)
    table.add_column("Capacity", justify="right", width=12)
    table.add_column("Utilization", width=22)
    table.add_column("Status", justify="center", width=10)

    rows = [
        ("Logic (LUT4)", lut_used, lut_cap),
        ("Registers (FF)", ff_used, ff_cap),
        ("Block RAM", bram_used, bram_cap),
        ("DSP (MULT)", dsp_used, dsp_cap),
    ]

    for name, used, cap in rows:
        if cap > 0:
            pct = (used / cap) * 100
            bar = format_progress_bar(used, cap)
            if pct > 100:
                status = "[bold red]OVER[/bold red]"
            elif pct > 85:
                status = "[bold yellow]WARN[/bold yellow]"
            else:
                status = "[bold green]OK[/bold green]"
            table.add_row(name, f"{used:,}", f"{cap:,}", bar, status)
        else:
            table.add_row(name, f"{used:,}", "N/A", f"{used:,}", "[green]OK[/green]")

    console.print("\n")
    console.print(table)

    # Timing and bitstream section
    f_target = timing_info.get("target_mhz", board_cfg.get("clk_freq", 27000000) / 1e6)
    f_max = timing_info.get("fmax_mhz")

    timing_text = Text()
    if f_max:
        slack = f_max - f_target
        if slack >= 0:
            slack_str = f"+{slack:.2f} MHz (Timing Constraints MET ✓)"
            slack_style = "bold green"
        else:
            slack_str = f"{slack:.2f} MHz (Timing Constraints VIOLATED ✗)"
            slack_style = "bold red"
        timing_text.append(f"Timing Report  : Target {f_target:.2f} MHz  |  Estimated Fmax : {f_max:.2f} MHz\n", style="bold")
        timing_text.append(f"Timing Slack   : ", style="bold")
        timing_text.append(f"{slack_str}\n", style=slack_style)
    else:
        timing_text.append(f"Timing Report  : Target {f_target:.2f} MHz (Static timing evaluated in PnR stage)\n")

    if bitstream_path and bitstream_path.exists():
        size_kib = bitstream_path.stat().st_size / 1024.0
        rel_bitstream = bitstream_path.relative_to(CLI_DIR.parent) if bitstream_path.is_relative_to(CLI_DIR.parent) else bitstream_path
        timing_text.append(f"Bitstream File : {rel_bitstream} ({size_kib:.1f} KiB)\n", style="bold cyan")

    panel = Panel(
        timing_text,
        title="[bold blue]Timing & Bitstream Summary[/bold blue]",
        border_style="blue",
        expand=False
    )
    console.print(panel)

    if stopped_at == "synth":
        console.print(f"[bold green][✓] Yosys synthesis completed in {duration:.1f}s. (Stopped at --synth-only / --yosys)[/bold green]\n")
    elif stopped_at == "pnr":
        console.print(f"[bold green][✓] Placement & Routing completed in {duration:.1f}s. (Stopped at --pnr-only / --nextpnr)[/bold green]\n")
    else:
        console.print(f"[bold green][✓] Synthesis, Placement & Routing completed in {duration:.1f}s. Ready to flash![/bold green]\n")

def _prepare_sources_and_constraints(
    src: Optional[Path],
    project_root: Path,
    hw_build_dir: Path,
    board: str,
    board_cfg: Dict[str, Any],
    clk_override: Optional[str],
    no_dsp: bool,
    top_name: Optional[str],
    cst_override: Optional[Path],
    log_file: Any,
) -> Tuple[Optional[List[Path]], Optional[str], Optional[Path]]:
    """Resolves source files, compiles .scala if needed, detects top module and adapts CST constraints."""
    if src is not None:
        src_p = src if src.is_absolute() else (project_root / src).resolve()
    else:
        rtl_cand = project_root / "rtl"
        verilog_cand = project_root / "verilog"
        if rtl_cand.exists() and list(rtl_cand.glob("*.v")):
            src_p = rtl_cand
        elif verilog_cand.exists() and list(verilog_cand.glob("*.v")):
            src_p = verilog_cand
        else:
            src_p = rtl_cand

    if src_p.is_file() and src_p.suffix == ".scala":
        console.print(Panel(
            f"[bold]Target Board :[/bold] {board_cfg['name']} ({board_cfg['fpga']})\n"
            f"[bold]Source Model :[/bold] {src_p.relative_to(project_root) if src_p.is_relative_to(project_root) else src_p} (.scala)\n"
            f"[bold]Output Dir   :[/bold] {hw_build_dir.relative_to(project_root) if hw_build_dir.is_relative_to(project_root) else hw_build_dir}",
            title="[bold cyan]spinalML Hardware Builder[/bold cyan]",
            border_style="cyan"
        ))
        console.print("\n[bold cyan] Step 0: Compiling Scala model to Verilog (turnkey UartSoC)...[/bold cyan]")
        from .cli import compile as compile_cmd
        rtl_dir = project_root / "rtl"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        for old_f in list(rtl_dir.glob("*.v")) + list(rtl_dir.glob("*.bin")):
            try:
                old_f.unlink()
            except Exception:
                pass
        try:
            import typer
            compile_cmd(
                file=src_p,
                out=rtl_dir,
                chain=True,
                soc=True,
                board=board,
                clk=clk_override,
                baud=None,
                out_count=None,
                word_width=None,
                bram_words=None,
                no_dsp=no_dsp
            )
        except typer.Exit as te:
            if te.exit_code != 0:
                console.print(f"[bold red]Scala compilation exited with code {te.exit_code}[/bold red]")
                return None, None, None
        except Exception as e:
            console.print(f"[bold red]Scala compilation failed: {e}[/bold red]")
            return None, None, None
        src_p = rtl_dir

    if src_p.is_dir():
        v_files = sorted(src_p.glob("*.v"))
    elif src_p.is_file() and src_p.suffix == ".v":
        v_files = [src_p]
    else:
        console.print(f"[bold red]Error: No Verilog files found at {src_p}[/bold red]")
        return None, None, None

    if not v_files:
        console.print(f"[bold red]Error: No .v files found in directory {src_p}[/bold red]")
        return None, None, None

    actual_top, ports = detect_top_module(v_files, top_name)

    final_clk_hz = parse_frequency(clk_override) if clk_override else board_cfg["clk_freq"]
    final_clk_mhz = final_clk_hz / 1_000_000.0

    console.print(Panel(
        f"[bold]Target Board :[/bold] {board_cfg['name']} ({board_cfg['fpga']})\n"
        f"[bold]Source RTL   :[/bold] {src_p.relative_to(project_root) if src_p.is_relative_to(project_root) else src_p} ({len(v_files)} Verilog file(s), top: [cyan]{actual_top}[/cyan])\n"
        f"[bold]Output Dir   :[/bold] {hw_build_dir.relative_to(project_root) if hw_build_dir.is_relative_to(project_root) else hw_build_dir}\n"
        f"[bold]Clock Target :[/bold] {final_clk_mhz:.2f} MHz",
        title="[bold cyan]spinalML Hardware Builder[/bold cyan]",
        border_style="cyan"
    ))

    resolved_cst = resolve_constraints_file(board_cfg, cst_override)
    adapted_cst = hw_build_dir / "pins.cst"
    adapt_constraints_for_ports(resolved_cst, ports, adapted_cst)

    return v_files, actual_top, adapted_cst


def _execute_yosys_synthesis(
    board_cfg: Dict[str, Any],
    actual_top: str,
    v_files: List[Path],
    hw_build_dir: Path,
    synth_json: Path,
    project_root: Path,
    eda_env: Dict[str, str],
    log_file: Any,
    no_dsp: bool,
) -> Tuple[int, Dict[str, int], float]:
    """Runs Yosys synthesis and returns (returncode, resources, duration)."""
    yosys_bin = get_bin_path("yosys")
    synth_cmd_name = board_cfg.get("build", {}).get("synth_cmd", "synth_gowin")

    if no_dsp or "synth_gowin" in synth_cmd_name:
        if "-nodsp" not in synth_cmd_name:
            synth_cmd_name = f"{synth_cmd_name} -nodsp"

    selected_v_files = filter_unique_verilog_files(v_files, actual_top)
    v_args = " ".join([f'"{str(vf).replace(chr(92), "/")}"' for vf in selected_v_files])
    yosys_script = f"read_verilog -sv {v_args}; {synth_cmd_name} -top {actual_top} -json \"{str(synth_json).replace(chr(92), '/')}\""

    yosys_args = [str(yosys_bin), "-p", yosys_script]
    log_file.write(f"Source Verilog files detected: {[f.name for f in v_files]}\n")
    log_file.write(f"Selected non-redundant files for Yosys: {[f.name for f in selected_v_files]}\n")
    log_file.write(f"\n--- Yosys Synthesis Command ---\n{' '.join(yosys_args)}\n\n")
    log_file.flush()

    pass_regex = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)*)\.\s+(Executing\s+.*)')
    lut_regex = re.compile(r'^\s*(?:LUT\d+|LUT)\s+(\d+)')
    dff_regex = re.compile(r'^\s*(?:DFF\w*|DFFE|DFFR)\s+(\d+)')

    resources: Dict[str, int] = {}
    t0 = time.time()
    current_status = "Elaborating RTL hierarchy..."

    with Live(Spinner("dots", text=Text(f"[Step 1/3] Yosys : {current_status} (0.0s)", style="bold yellow")), console=console, refresh_per_second=10) as live:
        proc = subprocess.Popen(
            yosys_args,
            cwd=str(project_root),
            env=eda_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            log_file.write(line)
            m_pass = pass_regex.search(line)
            if m_pass:
                pass_no = m_pass.group(1)
                pass_desc = m_pass.group(2).strip()
                current_status = f"Pass {pass_no} : {pass_desc}"

            m_lut = lut_regex.search(line)
            if m_lut:
                resources["LUT4"] = resources.get("LUT4", 0) + int(m_lut.group(1))
            m_dff = dff_regex.search(line)
            if m_dff:
                resources["FF"] = resources.get("FF", 0) + int(m_dff.group(1))

            elapsed = time.time() - t0
            live.update(Spinner("dots", text=Text(f"[Step 1/3] Yosys Synthesis : {current_status} ({elapsed:.1f}s)", style="bold yellow")))

        proc.wait()

    duration_synth = time.time() - t0
    if proc.returncode != 0:
        log_path = hw_build_dir / "build.log"
        console.print(f"[bold red]Yosys synthesis failed with returncode {proc.returncode}. See {log_path} for details.[/bold red]")
        return proc.returncode, resources, duration_synth

    console.print(f" [bold green]✓[/bold green] [Step 1/3] Yosys synthesis completed in {duration_synth:.1f}s.")
    resources = extract_yosys_resources(synth_json, actual_top)
    return 0, resources, duration_synth


def _execute_nextpnr(
    board_cfg: Dict[str, Any],
    synth_json: Path,
    hw_build_dir: Path,
    adapted_cst: Path,
    final_clk_mhz: float,
    project_root: Path,
    eda_env: Dict[str, str],
    log_file: Any,
) -> Tuple[int, Dict[str, int], Dict[str, Any], float]:
    """Runs nextpnr place & route and returns (returncode, resources, timing_info, duration)."""
    pnr_tool_name = board_cfg.get("build", {}).get("pnr_tool", "nextpnr-himbaechel")
    pnr_bin = get_bin_path(pnr_tool_name)
    pnr_json = hw_build_dir / "pnr.json"
    pnr_report_path = hw_build_dir / "pnr_report.json"

    pnr_args = [
        str(pnr_bin),
        "--json", str(synth_json),
        "--write", str(pnr_json),
        "--freq", str(final_clk_mhz),
        "--report", str(pnr_report_path)
    ]
    pnr_args.extend(board_cfg.get("build", {}).get("pnr_args", []))

    if adapted_cst.exists() and "himbaechel" in pnr_tool_name:
        pnr_args.extend(["--vopt", f"cst={str(adapted_cst)}"])

    log_file.write(f"\n--- nextpnr Command ---\n{' '.join(pnr_args)}\n\n")
    log_file.flush()

    resources: Dict[str, int] = {}
    timing_info: Dict[str, Any] = {"target_mhz": final_clk_mhz}
    t1 = time.time()
    pnr_status = "Packing logic clusters..."

    with Live(Spinner("dots", text=Text(f"[Step 2/3] nextpnr : {pnr_status} (0.0s)", style="bold yellow")), console=console, refresh_per_second=10) as live:
        proc = subprocess.Popen(
            pnr_args,
            cwd=str(project_root),
            env=eda_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        pnr_lut_regex = re.compile(r'^\s*Info:\s+LUT4:\s+(\d+)\s*/\s*(\d+)')
        pnr_dff_regex = re.compile(r'^\s*Info:\s+DFF[RE]*:\s+(\d+)\s*/\s*(\d+)')
        pnr_bsram_regex = re.compile(r'^\s*Info:\s+BSRAM:\s+(\d+)\s*/\s*(\d+)')
        pnr_dsp_regex = re.compile(r'^\s*Info:\s+MULT\w*:\s+(\d+)\s*/\s*(\d+)')
        fmax_regex = re.compile(r'Max frequency for clock.*:\s*([\d\.]+)\s*MHz')

        for line in proc.stdout:
            log_file.write(line)
            if "Packing" in line:
                pnr_status = "Packing logic clusters..."
            elif "Placing" in line:
                pnr_status = "Placing cells (Simulated Annealing)..."
            elif "Routing" in line:
                pnr_status = "Routing nets..."
            elif "Timing" in line:
                pnr_status = "Static timing analysis..."

            m_lut = pnr_lut_regex.search(line)
            if m_lut:
                resources["LUT4"] = int(m_lut.group(1))
            m_dff = pnr_dff_regex.search(line)
            if m_dff:
                resources["FF"] = int(m_dff.group(1))
            m_bsram = pnr_bsram_regex.search(line)
            if m_bsram:
                resources["BRAM"] = int(m_bsram.group(1))
            m_dsp = pnr_dsp_regex.search(line)
            if m_dsp:
                resources["DSP"] = int(m_dsp.group(1))

            m_fmax = fmax_regex.search(line)
            if m_fmax:
                timing_info["fmax_mhz"] = float(m_fmax.group(1))

            elapsed = time.time() - t1
            live.update(Spinner("dots", text=Text(f"[Step 2/3] nextpnr Place & Route : {pnr_status} ({elapsed:.1f}s)", style="bold yellow")))

        proc.wait()

    duration_pnr = time.time() - t1
    if proc.returncode != 0:
        log_path = hw_build_dir / "build.log"
        console.print(f"[bold red]nextpnr failed with returncode {proc.returncode}. See {log_path} for details.[/bold red]")
        return proc.returncode, resources, timing_info, duration_pnr

    pnr_res, fmax_achieved = extract_pnr_resources(pnr_report_path)
    if pnr_res:
        resources.update(pnr_res)
    if fmax_achieved is not None:
        timing_info["fmax_mhz"] = fmax_achieved

    console.print(f" [bold green]✓[/bold green] [Step 2/3] nextpnr Place & Route completed in {duration_pnr:.1f}s.")
    return 0, resources, timing_info, duration_pnr


def _execute_bitstream_packing(
    board_cfg: Dict[str, Any],
    pnr_json: Path,
    hw_build_dir: Path,
    project_root: Path,
    eda_env: Dict[str, str],
    log_file: Any,
) -> Tuple[int, Optional[Path], float]:
    """Packs bitstream using board pack tool and returns (returncode, bitstream_path, duration)."""
    pack_tool_name = board_cfg.get("build", {}).get("pack_tool", "gowin_pack")
    if pack_tool_name == "gowin_pack":
        ensure_apycula_patched()

    pack_bin = get_bin_path(pack_tool_name)
    bitstream_name = board_cfg.get("build", {}).get("bitstream_name", "top.fs")
    bitstream_path = hw_build_dir / bitstream_name

    pack_args = [str(pack_bin)]
    pack_args.extend(board_cfg.get("build", {}).get("pack_args", []))
    pack_args.extend(["-o", str(bitstream_path), str(pnr_json)])

    log_file.write(f"\n--- Bitstream Pack Command ---\n{' '.join(pack_args)}\n\n")
    log_file.flush()

    t2 = time.time()
    with Live(Spinner("dots", text=Text(f"[Step 3/3] Bitstream packaging into {bitstream_name}... (0.0s)", style="bold yellow")), console=console, refresh_per_second=10) as live:
        proc = subprocess.Popen(
            pack_args,
            cwd=str(project_root),
            env=eda_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            log_file.write(line)
            elapsed = time.time() - t2
            live.update(Spinner("dots", text=Text(f"[Step 3/3] Bitstream packaging into {bitstream_name}... ({elapsed:.1f}s)", style="bold yellow")))
        proc.wait()

    duration_pack = time.time() - t2
    if proc.returncode != 0:
        log_path = hw_build_dir / "build.log"
        console.print(f"[bold red]Bitstream packing failed with returncode {proc.returncode}. See {log_path} for details.[/bold red]")
        return proc.returncode, None, duration_pack

    console.print(f" [bold green]✓[/bold green] [Step 3/3] Bitstream {bitstream_name} packaged successfully in {duration_pack:.1f}s.")
    return 0, bitstream_path, duration_pack


def run_build(
    src: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    board: str = "tang-primer-20k",
    cst_override: Optional[Path] = None,
    top_name: Optional[str] = None,
    synth_only: bool = False,
    pnr_only: bool = False,
    clk_override: Optional[str] = None,
    no_dsp: bool = False
) -> int:
    """
    Orchestrates the entire hardware build pipeline:
    1. Optional compilation of .scala model -> Verilog
    2. Yosys RTL Elaboration & Technology Mapping
    3. nextpnr Placement & Routing
    4. Bitstream Packaging (gowin_pack)
    """
    project_root = CLI_DIR.parent
    start_total_time = time.time()

    # 1. Load board configuration
    try:
        board_cfg = load_board_config(board)
    except Exception as e:
        console.print(f"[bold red]Error loading board profile '{board}': {e}[/bold red]")
        return 1

    # 2. Resolve clock frequency & output directories
    final_clk_hz = parse_frequency(clk_override) if clk_override else board_cfg["clk_freq"]
    final_clk_mhz = final_clk_hz / 1_000_000.0

    board_slug = board_cfg["file_path"].stem
    if out_dir:
        hw_build_dir = out_dir if out_dir.is_absolute() else (project_root / out_dir).resolve()
    else:
        hw_build_dir = project_root / "hw_build" / board_slug
    hw_build_dir.mkdir(parents=True, exist_ok=True)

    log_path = hw_build_dir / "build.log"
    log_file = open(log_path, "w", encoding="utf-8")

    # 3. Source & constraint preparation
    v_files, actual_top, adapted_cst = _prepare_sources_and_constraints(
        src=src,
        project_root=project_root,
        hw_build_dir=hw_build_dir,
        board=board,
        board_cfg=board_cfg,
        clk_override=clk_override,
        no_dsp=no_dsp,
        top_name=top_name,
        cst_override=cst_override,
        log_file=log_file
    )
    if v_files is None or actual_top is None or adapted_cst is None:
        log_file.close()
        return 1

    eda_env = get_eda_env()
    synth_json = hw_build_dir / "synth.json"
    timing_info: Dict[str, Any] = {"target_mhz": final_clk_mhz}

    # 4. Phase 1: Yosys Synthesis
    rc_synth, resources, duration_synth = _execute_yosys_synthesis(
        board_cfg=board_cfg,
        actual_top=actual_top,
        v_files=v_files,
        hw_build_dir=hw_build_dir,
        synth_json=synth_json,
        project_root=project_root,
        eda_env=eda_env,
        log_file=log_file,
        no_dsp=no_dsp
    )
    if rc_synth != 0:
        log_file.close()
        return rc_synth

    if synth_only:
        log_file.close()
        render_final_report(board_cfg, actual_top, resources, timing_info, None, duration_synth, stopped_at="synth")
        return 0

    # 5. Phase 2: nextpnr Place & Route
    rc_pnr, pnr_res, pnr_timing, duration_pnr = _execute_nextpnr(
        board_cfg=board_cfg,
        synth_json=synth_json,
        hw_build_dir=hw_build_dir,
        adapted_cst=adapted_cst,
        final_clk_mhz=final_clk_mhz,
        project_root=project_root,
        eda_env=eda_env,
        log_file=log_file
    )
    if rc_pnr != 0:
        log_file.close()
        return rc_pnr

    resources.update(pnr_res)
    timing_info.update(pnr_timing)

    if pnr_only:
        log_file.close()
        total_d = time.time() - start_total_time
        render_final_report(board_cfg, actual_top, resources, timing_info, None, total_d, stopped_at="pnr")
        return 0

    # 6. Phase 3: Bitstream Packaging
    pnr_json = hw_build_dir / "pnr.json"
    rc_pack, bitstream_path, _ = _execute_bitstream_packing(
        board_cfg=board_cfg,
        pnr_json=pnr_json,
        hw_build_dir=hw_build_dir,
        project_root=project_root,
        eda_env=eda_env,
        log_file=log_file
    )
    log_file.close()

    if rc_pack != 0:
        return rc_pack

    total_duration = time.time() - start_total_time
    render_final_report(board_cfg, actual_top, resources, timing_info, bitstream_path, total_duration, stopped_at="full")
    return 0
