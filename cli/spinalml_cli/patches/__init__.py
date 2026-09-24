# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Upstream toolchain backports (Gowin / oss-cad-suite).

Each patch here works around a known upstream tool bug and is version-gated:
it applies only when the bundled toolchain actually needs it, reports what it
did (or why it skipped), and supports dry-run. Delete a patch in ONE place
once the pinned tools are new enough (each module states its removal condition).
"""

from pathlib import Path
from typing import Dict, Optional

from .apycula import ensure_apycula_patched
from .nextpnr_cin import patch_gowin_cin_from_logic


def apply_toolchain_patches(pnr_json: Optional[Path] = None, dry_run: bool = False,
                            debug: bool = False, console=None) -> Dict[str, object]:
    """Applies needed toolchain patches. pnr_json=None skips the pnr.json backport.

    Returns a report dict: {"apycula": <status str>, "cin_patched": <int>}.
    """
    report: Dict[str, object] = {}
    report["apycula"] = ensure_apycula_patched(dry_run=dry_run, debug=debug, console=console)
    n_cin = 0
    if pnr_json is not None:
        n_cin = patch_gowin_cin_from_logic(pnr_json, dry_run=dry_run)
        if n_cin and not dry_run:
            msg = (f"Patched {n_cin} CIN-from-logic head ALU(s) "
                   f"(nextpnr 6030081a15 backport).")
            if debug or console is None:
                print(msg)
            else:
                console.print(f" [yellow]{msg}[/yellow]")
    report["cin_patched"] = n_cin
    if dry_run:
        msg = f"[dry-run] toolchain patches report: {report}"
        if debug or console is None:
            print(msg)
        else:
            console.print(f"[dim]{msg}[/dim]")
    return report
