# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""nextpnr 6030081a15 backport ("gowin: fix carry-in adapter table for CIN from logic").

The head ALU inserted for a carry chain whose CIN comes from logic takes
that signal on I0 and leaves CIN unconnected. The buggy table made it
output COUT = I0 | CIN, so the injected carry depended on the carry state
at the *placed* location (neighbouring active carry chain); 0x000a gives
COUT = I0. The latent corruption only shows up for the layouts unlucky
enough to chain such a head ALU onto an active carry chain (observed as
saturated MNIST logits on the RNE + explicit DSP build).

Applied to the routed pnr.json just before gowin_pack. Content-driven:
only cells matching the buggy pattern are touched, so it is a natural no-op
once nextpnr is updated. REMOVE this module once the pinned oss-cad-suite
ships the fix (check nextpnr-himbaechel --version / changelog).
See docs/bugs/2026-09-gowin-rne-dsp-lut-saturation.md §8.
"""

import json
from pathlib import Path

BAD16 = "0101000001011010"   # RAW_ALU_LUT 0x505a
GOOD16 = "0000000000001010"  # RAW_ALU_LUT 0x000a


def patch_gowin_cin_from_logic(pnr_json: Path, dry_run: bool = False) -> int:
    """Rewrites buggy head-ALU LUTs in the routed pnr.json. Returns patch count."""
    try:
        with open(pnr_json, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    patched = 0
    for mod in data.get("modules", {}).values():
        for cell in mod.get("cells", {}).values():
            if cell.get("type") != "ALU":
                continue
            parms = cell.get("parameters", {})
            if parms.get("CIN_NETTYPE") != "LOGIC":
                continue
            raw = parms.get("RAW_ALU_LUT", "")
            if len(raw) >= 16 and raw[-16:] == BAD16:
                if not dry_run:
                    parms["RAW_ALU_LUT"] = raw[:-16] + GOOD16
                patched += 1
    if patched and not dry_run:
        with open(pnr_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return patched
