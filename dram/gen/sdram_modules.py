"""DDR3 SDRAM module definitions missing from upstream LiteDRAM.

Upstream ``litedram.modules`` covers many Micron/ISSI parts but not the
SK Hynix chip fitted on our Tang Primer 20K revision, so the definition
lives here (same ``DDR3Module`` conventions, no LiteDRAM fork needed).
"""

from litedram.modules import DDR3Module, _TechnologyTimings, _SpeedgradeTimings


class H5TC1G63EFR(DDR3Module):
    """SK Hynix 1Gb DDR3 SDRAM, 64M x 16 organisation (128MiB).

    Geometry matches ``MT41K64M16`` (8 banks, 8192 rows, 1024 cols).
    Timings are conservative 1Gb DDR3 values: the core runs far below the
    speed grade (54MHz DDR clock at 27MHz sys), so every ns constraint
    quantizes to a couple of cycles with wide margin.
    """
    # Geometry -----------------------------------------------------------------
    nbanks = 8
    nrows  = 8192
    ncols  = 1024
    # Timings (conservative 1Gb DDR3-1066 grade) -------------------------------
    # JEDEC DDR3 x16 organisation requires the 6 nCK tRRD floor (vs 4 nCK for x4/x8).
    technology_timings = _TechnologyTimings(
        tREFI=64e6/8192, tWTR=(4, 7.5), tCCD=(4, None), tRRD=(6, 10), tZQCS=(64, 80))
    speedgrade_timings = {
        "1066": _SpeedgradeTimings(
            tRP=13.1, tRCD=13.1, tWR=13.1, tRFC=(90, None), tFAW=(None, 50), tRAS=37.5),
    }
    speedgrade_timings["default"] = speedgrade_timings["1066"]
