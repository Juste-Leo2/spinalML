# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""apycula gowin_pack.py backport (B-multiplier IREG offsets).

Upstream bug: B multiplier attribute indices miss an offset of 2, causing
KeyError on IRBY_IREG0BL_0. Fixed upstream; REMOVE this module once the pinned
oss-cad-suite predates no supported install anymore (i.e. every bundled
gowin_pack.py already contains the r_offset logic).

Predicate (content-driven, no hardcoded interpreter versions):
- old (buggy) snippet present  -> patch needed (or dry-run report).
- already-patched snippet present -> skip, log "already patched".
- neither present (unknown upstream version) -> loud warning + skip, never blind-write.
"""

from pathlib import Path
from typing import List

from ..config import TOOLS_DIR

OLD_TARGET = """            else:
                if is_even:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}H_{pair_idx * 4 + 1}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}H_{pair_idx * 4 + 1}', "ENABLE"))
                else:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}L_{pair_idx * 4}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}L_{pair_idx * 4}', "ENABLE"))"""

NEW_REPLACEMENT = """            else:
                r_offset = 0 if r == 'A' else 2
                if is_even:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}H_{pair_idx * 4 + 1 + r_offset}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}H_{pair_idx * 4 + 1 + r_offset}', "ENABLE"))
                else:
                    attr_vals.append(AttrVal(f'IRBY_IREG{pair_idx}{r}L_{pair_idx * 4 + r_offset}', "ENABLE"))
                    attr_vals.append(AttrVal(f'IRNS_IREG{pair_idx}{r}L_{pair_idx * 4 + r_offset}', "ENABLE"))"""


def find_apycula_files() -> List[Path]:
    """Locates every bundled apycula/gowin_pack.py (any vendored python version)."""
    suite_lib = TOOLS_DIR / "oss-cad-suite" / "lib"
    if not suite_lib.is_dir():
        return []
    return sorted(suite_lib.glob("*/site-packages/apycula/gowin_pack.py"))


def ensure_apycula_patched(dry_run: bool = False, debug: bool = False,
                           console=None) -> str:
    """Applies the backport where needed. Returns a status string for logs.

    Statuses: "not-found" (no suite installed), "patched" (fixed now),
    "already-patched", "unknown-version" (loud warning, skipped).
    With dry_run=True, reports without writing.
    """
    files = find_apycula_files()
    if not files:
        return "not-found"

    def say(msg: str, warn: bool = False):
        if debug or console is None:
            print(msg)
        elif console:
            console.print(f"[yellow]{msg}[/yellow]" if warn else f"[dim]{msg}[/dim]")

    statuses = []
    for apycula_file in files:
        try:
            content = apycula_file.read_text(encoding="utf-8")
        except Exception as e:
            statuses.append(f"{apycula_file}: unreadable ({e})")
            continue
        if NEW_REPLACEMENT in content:
            statuses.append(f"{apycula_file}: already-patched")
            continue
        if OLD_TARGET not in content:
            say(f"WARNING: {apycula_file} matches neither the buggy nor the fixed "
                f"apycula version; skipping blind write.", warn=True)
            statuses.append(f"{apycula_file}: unknown-version")
            continue
        if dry_run:
            statuses.append(f"{apycula_file}: would-patch")
            continue
        apycula_file.write_text(content.replace(OLD_TARGET, NEW_REPLACEMENT, 1),
                                 encoding="utf-8")
        say(f"Patched {apycula_file} (apycula B-multiplier IREG offsets).")
        statuses.append(f"{apycula_file}: patched")
    if any(s.endswith(": patched") or s.endswith(": would-patch") for s in statuses):
        return "patched"
    if any(s.endswith("unknown-version") for s in statuses):
        return "unknown-version"
    return "already-patched"
