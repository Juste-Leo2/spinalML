# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import json
import os
import sys
import platform
from pathlib import Path

# Paths & PyInstaller detection
IS_FROZEN = getattr(sys, "frozen", False)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent.resolve()))
CLI_DIR = Path(__file__).parent.parent.resolve()
TOOLS_DIR = Path.home() / ".spinalml_tools"

def get_project_root() -> Path:
    """
    Returns the active project/workspace root directory.
    Searches upwards from current working directory for repository markers (build.mill, .git, spinalML),
    falling back to CLI_DIR.parent if running from source, or Path.cwd().
    """
    curr = Path.cwd().resolve()
    for p in [curr] + list(curr.parents):
        if (p / "build.mill").exists() or (p / ".git").exists() or (p / "spinalML").is_dir():
            return p
    if not IS_FROZEN and (CLI_DIR.parent / "build.mill").exists():
        return CLI_DIR.parent.resolve()
    return curr

def get_bundled_scaffold_dir() -> Path:
    """Returns the bundled framework directory containing spinalML and build.mill."""
    if IS_FROZEN:
        return BUNDLE_DIR
    return CLI_DIR.parent

def get_active_framework_root() -> Path:
    """
    Returns the root directory where build.mill and spinalML/ are located.
    If running inside the spinalML repository, returns that repository.
    If running standalone outside any repository, seeds ~/.spinalml_tools/framework
    from the bundled assets to provide a persistent, warm Mill build workspace.
    """
    root = get_project_root()
    if (root / "spinalML").is_dir() and (root / "build.mill").is_file():
        return root

    scaffold = TOOLS_DIR / "framework"
    bundled = get_bundled_scaffold_dir()
    if not (scaffold / "build.mill").exists() and (bundled / "build.mill").exists():
        scaffold.mkdir(parents=True, exist_ok=True)
        import shutil
        if (bundled / "build.mill").exists():
            shutil.copy2(bundled / "build.mill", scaffold / "build.mill")
        if (bundled / "spinalML").exists():
            if (scaffold / "spinalML").exists():
                shutil.rmtree(scaffold / "spinalML")
            shutil.copytree(bundled / "spinalML", scaffold / "spinalML")
        if (bundled / "boards").exists() and not (scaffold / "boards").exists():
            shutil.copytree(bundled / "boards", scaffold / "boards")

    if (scaffold / "build.mill").exists():
        return scaffold
    return root


def load_config() -> dict:
    candidates = [
        BUNDLE_DIR / "config.json",
        BUNDLE_DIR / "cli" / "config.json",
        CLI_DIR / "config.json",
        get_project_root() / "cli" / "config.json"
    ]
    for cfg in candidates:
        if cfg.exists():
            with open(cfg, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Config file not found. Checked: {candidates}")


def get_os_arch() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return "windows-x64" # oss-cad-suite only provides x64 for windows
    elif system == "linux":
        if "arm" in machine or "aarch64" in machine:
            return "linux-arm64"
        return "linux-x64"
    elif system == "darwin":
        if "arm" in machine or "aarch64" in machine:
            return "darwin-arm64"
        return "darwin-x64"
    
    raise ValueError(f"Unsupported OS/Arch combination: {system}/{machine}")

def get_mill_url(config: dict) -> str:
    system = platform.system().lower()
    if system == "windows":
        return config["tools"]["mill"]["bat"]
    return config["tools"]["mill"]["sh"]

def get_oss_cad_suite_url(config: dict) -> str:
    os_arch = get_os_arch()
    urls = config["tools"]["oss-cad-suite"]
    if os_arch not in urls:
        raise ValueError(f"No OSS CAD Suite build found for {os_arch}")
    return urls[os_arch]

def get_w64devkit_url(config: dict) -> str:
    os_arch = get_os_arch()
    urls = config["tools"].get("w64devkit", {})
    if os_arch not in urls:
        raise ValueError(f"No w64devkit build found for {os_arch}")
    return urls[os_arch]

def get_bin_path(tool_name: str) -> Path:
    """Returns the absolute path to a tool's executable."""
    import shutil
    system = platform.system().lower()
    is_win = system == "windows"
    
    if tool_name == "mill":
        tool_p = TOOLS_DIR / ("mill.bat" if is_win else "mill")
        if tool_p.exists():
            return tool_p
        which = shutil.which("mill.bat" if is_win else "mill") or shutil.which("mill")
        if which:
            return Path(which)
        return tool_p
    else:
        bin_dir = TOOLS_DIR / "oss-cad-suite" / "bin"
        if is_win:
            # Try exact .exe
            exe_path = bin_dir / f"{tool_name}.exe"
            if exe_path.exists():
                return exe_path
            # Try _bin.exe (for verilator)
            bin_exe = bin_dir / f"{tool_name}_bin.exe"
            if bin_exe.exists():
                return bin_exe
            which = shutil.which(f"{tool_name}.exe") or shutil.which(f"{tool_name}_bin.exe") or shutil.which(tool_name)
            if which:
                return Path(which)
            return bin_dir / tool_name
        else:
            p = bin_dir / tool_name
            if p.exists():
                return p
            which = shutil.which(tool_name)
            if which:
                return Path(which)
            return p
