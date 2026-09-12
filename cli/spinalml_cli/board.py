# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .config import CLI_DIR, BUNDLE_DIR, get_project_root

def get_boards_dir() -> Path:
    """Returns the path to the boards/ directory (active project first, bundled fallback)."""
    local_boards = get_project_root() / "boards"
    if local_boards.exists():
        return local_boards
    bundled_boards = BUNDLE_DIR / "boards"
    if bundled_boards.exists():
        return bundled_boards
    return CLI_DIR.parent / "boards"


def list_available_boards() -> List[str]:
    """Returns a list of all board profile names found in boards/*.json."""
    boards_dir = get_boards_dir()
    if not boards_dir.exists():
        return []
    return sorted([p.stem for p in boards_dir.glob("*.json")])

def parse_frequency(freq: Union[str, int, float]) -> int:
    """
    Parses a frequency value into Hertz (integer).
    Supports strings with units like '27MHz', '50MHz', '100MHz', '12MHz', '100kHz'
    or plain numeric strings / integers.
    """
    if isinstance(freq, (int, float)):
        return int(freq)
        
    s = str(freq).strip()
    m = re.match(r"^([\d\.]+)\s*([a-zA-Z]+)?$", s)
    if not m:
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"Invalid frequency format: '{freq}'. Expected e.g. '27MHz', '50MHz', '27000000'.")
            
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    
    if unit in ("GHZ", "G"):
        return int(val * 1_000_000_000)
    elif unit in ("MHZ", "M"):
        return int(val * 1_000_000)
    elif unit in ("KHZ", "K"):
        return int(val * 1_000)
    elif unit in ("HZ", ""):
        return int(val)
    else:
        raise ValueError(f"Unknown frequency unit '{unit}' in '{freq}'. Use Hz, kHz, MHz, or GHz.")

def load_board_config(board_name_or_path: str) -> Dict[str, Any]:
    """
    Loads a board configuration JSON from boards/<name>.json or a direct file path.
    Returns a dictionary of board parameters with fallback defaults.
    """
    boards_dir = get_boards_dir()
    candidate_path = Path(board_name_or_path)

    # 1. Direct file path
    if candidate_path.is_file():
        target_file = candidate_path
    # 2. boards/<name>.json
    elif (boards_dir / f"{board_name_or_path}.json").is_file():
        target_file = boards_dir / f"{board_name_or_path}.json"
    # 3. boards/<name> (if extension provided)
    elif (boards_dir / board_name_or_path).is_file():
        target_file = boards_dir / board_name_or_path
    else:
        avail = list_available_boards()
        avail_str = f" Available boards: {', '.join(avail)}" if avail else " No boards found in boards/."
        raise FileNotFoundError(f"Board profile '{board_name_or_path}' not found.{avail_str}")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read board config file {target_file}: {e}")

    # Standardize and supply defaults
    project_root = get_project_root()
    raw_build = data.get("build", {})
    raw_flash = data.get("flash", {})
    raw_limits = data.get("limits", {})

    default_cst_rel = raw_build.get("default_cst")
    default_cst_path = None
    if default_cst_rel:
        if (project_root / default_cst_rel).exists():
            default_cst_path = (project_root / default_cst_rel).resolve()
        elif (BUNDLE_DIR / default_cst_rel).exists():
            default_cst_path = (BUNDLE_DIR / default_cst_rel).resolve()
        else:
            default_cst_path = (project_root / default_cst_rel).resolve()


    config = {
        "name": data.get("name", target_file.stem),
        "vendor": data.get("vendor", "Unknown"),
        "family": data.get("family", ""),
        "fpga": data.get("fpga", ""),
        "clk_freq": parse_frequency(data.get("clk_freq", 27000000)),
        "baud_rate": int(data.get("baud_rate", 115200)),
        "bram_words": int(data.get("bram_words", 4096)),
        "description": data.get("description", ""),
        "file_path": target_file,
        "build": {
            "synth_cmd": raw_build.get("synth_cmd", "synth_gowin" if data.get("vendor") == "Gowin" else "synth"),
            "pnr_tool": raw_build.get("pnr_tool", "nextpnr-himbaechel"),
            "pnr_args": raw_build.get("pnr_args", []),
            "default_cst": default_cst_path,
            "pack_tool": raw_build.get("pack_tool", "gowin_pack"),
            "pack_args": raw_build.get("pack_args", []),
            "bitstream_name": raw_build.get("bitstream_name", "top.fs" if data.get("vendor") == "Gowin" else "top.bit")
        },
        "flash": {
            "programmer": raw_flash.get("programmer", "openFPGALoader"),
            "board_target": raw_flash.get("board_target", target_file.stem.replace("-", ""))
        },
        "limits": {
            "lut": int(raw_limits.get("lut", 0)),
            "ff": int(raw_limits.get("ff", 0)),
            "bram": int(raw_limits.get("bram", 0)),
            "dsp": int(raw_limits.get("dsp", 0))
        }
    }
    return config

def resolve_constraints_file(board_cfg: Dict[str, Any], override_cst: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Resolves physical constraints file:
    1. override_cst if explicitly specified and exists
    2. board_cfg["build"]["default_cst"] if exists
    3. None if no constraints file found
    """
    if override_cst:
        p = Path(override_cst)
        if not p.is_absolute():
            p = get_project_root() / p
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Specified constraints file not found: {override_cst}")

    cst_candidate = board_cfg.get("build", {}).get("default_cst")
    if cst_candidate and Path(cst_candidate).exists():
        return Path(cst_candidate).resolve()
    return None

def detect_model_parameters(scala_file: Path) -> Dict[str, Any]:
    """
    Introspects a Scala model file to detect model parameters:
    - outCount: via outFeatures or outCount
    - wordWidth: via dataWidth in Axi4Config
    """
    detected: Dict[str, Any] = {}
    if not scala_file.exists():
        return detected

    try:
        content = scala_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return detected

    # Detect outFeatures (e.g. Linear(..., outFeatures = 10))
    m_out = re.search(r'outFeatures\s*=\s*(\d+)', content)
    if not m_out:
        m_out = re.search(r'outCount\s*=\s*(\d+)', content)
    if m_out:
        detected["out_count"] = int(m_out.group(1))

    # Detect AXI dataWidth (e.g. dataWidth = 64 or dataWidth = 32)
    m_data = re.search(r'dataWidth\s*=\s*(\d+)', content)
    if m_data:
        detected["word_width"] = int(m_data.group(1))

    return detected
