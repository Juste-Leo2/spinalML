#!/usr/bin/env python3
"""Validation for DRAM board configs (dram/configs/*.yml).

Checks geometry/timing consistency before generation: required keys,
known SDRAM module, 64-bit AXI width contract with the SpinalHDL
``DdrAdapter``, and expected density (bytes from geometry) against the
board profile. Exits non-zero with a loud message on failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdram_modules import H5TC1G63EFR  # noqa: F401 (keeps module importable)
from gowin_gen import SDRAM_MODULES

REQUIRED_KEYS = ("sys_clk_freq", "device", "devicename", "sdram_module")
AXI_DATA_WIDTH = 64


def validate(config, source="<config>"):
    errors = []
    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append("missing required key '%s'" % key)
    if errors:
        return errors
    if config["sdram_module"] not in SDRAM_MODULES:
        errors.append("unknown sdram_module '%s' (known: %s)" % (
            config["sdram_module"], sorted(SDRAM_MODULES)))
        return errors
    try:
        sys_clk = float(config["sys_clk_freq"])
    except (TypeError, ValueError):
        return ["sys_clk_freq '%s' is not a number" % (config["sys_clk_freq"],)]
    if not 1e6 <= sys_clk <= 200e6:
        errors.append("sys_clk_freq %g Hz outside sane range [1MHz, 200MHz]" % sys_clk)
    module = SDRAM_MODULES[config["sdram_module"]](sys_clk, "1:2")
    # x16 chips carry 2 bytes per column address.
    size_bytes = module.nbanks * module.nrows * module.ncols * 2
    expected = config.get("size_bytes")
    if expected is not None and int(expected) != size_bytes:
        errors.append("geometry gives %d bytes but size_bytes=%s" % (size_bytes, expected))
    if int(config.get("axi_data_width", AXI_DATA_WIDTH)) != AXI_DATA_WIDTH:
        errors.append("axi_data_width must be %d (DdrAdapter contract)" % AXI_DATA_WIDTH)
    return errors


def main():
    import yaml
    ok = True
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        errors = validate(config, source=path)
        if errors:
            ok = False
            for e in errors:
                print("%s: ERROR: %s" % (path, e))
        else:
            print("%s: OK" % path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
