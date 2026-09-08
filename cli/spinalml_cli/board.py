# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .config import CLI_DIR

def get_boards_dir() -> Path:
    """Returns the path to the boards/ directory at the project root."""
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
    config = {
        "name": data.get("name", target_file.stem),
        "vendor": data.get("vendor", "Unknown"),
        "family": data.get("family", ""),
        "fpga": data.get("fpga", ""),
        "clk_freq": parse_frequency(data.get("clk_freq", 27000000)),
        "baud_rate": int(data.get("baud_rate", 115200)),
        "bram_words": int(data.get("bram_words", 4096)),
        "description": data.get("description", ""),
        "file_path": target_file
    }
    return config

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
