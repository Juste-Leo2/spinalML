# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import CLI_DIR, TOOLS_DIR, get_bin_path
from .board import load_board_config
from .build_runner import get_eda_env

console = Console()

def run_flash(
    bitstream: Optional[Path] = None,
    board: str = "tang-primer-20k",
    sram: bool = True,
    flash_mem: bool = False
) -> int:
    """
    Programs the FPGA using openFPGALoader:
    - Auto-resolves bitstream from hw_build/<board>/<bitstream_name> if not specified.
    - Default mode is SRAM (fast volatile, -m).
    - Optional mode is Flash (SPI non-volatile, -f).
    """
    project_root = CLI_DIR.parent

    # 1. Load board configuration
    try:
        board_cfg = load_board_config(board)
    except Exception as e:
        console.print(f"[bold red]Error loading board profile '{board}': {e}[/bold red]")
        return 1

    board_slug = board_cfg["file_path"].stem
    flash_cfg = board_cfg.get("flash", {})
    prog_tool = flash_cfg.get("programmer", "openFPGALoader")
    board_target = flash_cfg.get("board_target", board_slug.replace("-", ""))

    # 2. Resolve bitstream file
    target_bitstream: Optional[Path] = None
    if bitstream:
        candidate = bitstream if bitstream.is_absolute() else (project_root / bitstream).resolve()
        if candidate.exists():
            target_bitstream = candidate
        else:
            console.print(f"[bold red]Error: Specified bitstream file does not exist: {candidate}[/bold red]")
            return 1
    else:
        bitstream_name = board_cfg.get("build", {}).get("bitstream_name", "top.fs")
        default_path = project_root / "hw_build" / board_slug / bitstream_name
        if default_path.exists():
            target_bitstream = default_path
        else:
            # Check legacy build/ directory
            legacy_path = project_root / "build" / board_slug / bitstream_name
            if legacy_path.exists():
                target_bitstream = legacy_path
            else:
                console.print(f"[bold red]Error: No bitstream found at {default_path}[/bold red]")
                console.print(f"[yellow]Please run 'spinalml build --board {board}' first to generate the bitstream.[/yellow]")
                return 1

    # 3. Determine flash mode (SRAM vs SPI Flash)
    if flash_mem:
        mode_flag = "-f"
        target_mode_desc = "SPI Flash (Non-Volatile / Persistent)"
    else:
        mode_flag = "-m"
        target_mode_desc = "SRAM (Volatile / Fast Load)"

    size_kib = target_bitstream.stat().st_size / 1024.0
    rel_bitstream = target_bitstream.relative_to(project_root) if target_bitstream.is_relative_to(project_root) else target_bitstream

    # Display programmer panel
    info_text = Text()
    info_text.append("Programmer : ", style="bold")
    info_text.append(f"{prog_tool} (JTAG / USB)\n", style="cyan")
    info_text.append("Board      : ", style="bold")
    info_text.append(f"{board_cfg['name']} ({board_target})\n", style="cyan")
    info_text.append("Target     : ", style="bold")
    info_text.append(f"{target_mode_desc}\n", style="magenta")
    info_text.append("File       : ", style="bold")
    info_text.append(f"{rel_bitstream} ({size_kib:.1f} KiB)\n", style="green")

    console.print(Panel(info_text, title="[bold cyan]spinalML Device Programmer[/bold cyan]", border_style="cyan"))

    prog_bin = get_bin_path(prog_tool)
    cmd = [str(prog_bin), "-b", board_target, mode_flag, str(target_bitstream)]

    eda_env = get_eda_env()
    t0 = time.time()

    try:
        console.print(f"[bold yellow]Connecting to probe and flashing {target_bitstream.name}...[/bold yellow]")
        result = subprocess.run(cmd, cwd=str(project_root), env=eda_env)
        elapsed = time.time() - t0

        if result.returncode == 0:
            console.print(f"\n[bold green][✓] FPGA configured successfully in {elapsed:.1f}s! Board is live.[/bold green]")
            return 0
        else:
            console.print(f"\n[bold red][✗] Programming failed with returncode {result.returncode}.[/bold red]")
            console.print("[yellow]Hint: Ensure your board USB cable is connected and probe permissions/drivers are installed.[/yellow]")
            return result.returncode
    except KeyboardInterrupt:
        console.print("\n[yellow]Programming interrupted by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error executing {prog_tool}: {e}[/bold red]")
        return 1
