import json
import os
import platform
from pathlib import Path

# Paths
CLI_DIR = Path(__file__).parent.parent.resolve()
CONFIG_FILE = CLI_DIR / "config.json"
TOOLS_DIR = Path.home() / ".spinalml_tools"

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

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
