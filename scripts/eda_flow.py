#!/usr/bin/env python3
# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Headless Gowin EDA factory: .scala --dram -> .fs, zero GUI clicks.

Stages:
  1. compile MODEL --dram          (rtl/DramSoCTop.v + dram/out/litedram_core.v)
  2. stage files -> WORKDIR/src
  3. write build.tcl (.gprj equivalent) + run gw_sh (syn + PnR + bitstream)
  4. report resources (PASS/FAIL vs expected envelope)
  5. optional --flash via programmer_cli (vendor driver, no Zadig dance)
  6. optional --validate via examples/UniversalScale/silicon_validate.py

Run on Windows with the CLI env (same link as everything else)::

    .\\.venv\\Scripts\\python.exe scripts\\eda_flow.py \\
        --model tests\\universal\\UniversalScaleDemo.scala [--flash] [--validate]
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BOARDS = {
    # board profile -> (part number, device version, hw_build cst dir)
    "tang-primer-20k": ("GW2A-LV18PG256C8/I7", "C", "tang-primer-20k"),
}

GOWIN_SH = Path(r"C:\Gowin\Gowin_V1.9.11.03_Education_x64\IDE\bin\gw_sh.exe")
PROGRAMMER_CLI = Path(
    r"C:\Gowin\Gowin_V1.9.11.03_Education_x64\Programmer\bin\programmer_cli.exe")

EXPECTED = {  # Proven 27/09 runs (8360 LUT / 4765 reg / DLL 1 / DQS 2).
    "lut_min": 7000, "lut_max": 10000,
    "reg_min": 4000, "reg_max": 5500,
    "bsram_max": 46,
}


def sh(cmd, cwd, tag):
    print(f"[{tag}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace")
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[{tag}] FAILED rc={rc}\n" + "".join(tail[-30:]))
    return rc


def main():
    ap = argparse.ArgumentParser(description="Headless Gowin EDA factory")
    ap.add_argument("--model", default=r"tests\universal\UniversalScaleDemo.scala")
    ap.add_argument("--board", default="tang-primer-20k")
    ap.add_argument("--workdir", default=r"E:\eda-factory\DRAM")
    ap.add_argument("--top", default="DramSoCTop")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip compile+gw_sh, only parse WORKDIR outputs")
    ap.add_argument("--flash", choices=["none", "programmer", "openfpgaloader"],
                    default="none")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    part, devver, cstdir = BOARDS[args.board]
    work = Path(args.workdir)
    src = work / "src"
    src.mkdir(parents=True, exist_ok=True)

    # 1. compile MODEL --dram ---------------------------------------------
    if not args.skip_compile and not args.report_only:
        sh([sys.executable, "cli/main.py", "compile", args.model,
            "--dram", "--board", args.board], REPO, "compile")

    # 2. stage -------------------------------------------------------------
    if not args.report_only:
        files = {
            REPO / "rtl" / "DramSoCTop.v": src / "DramSoCTop.v",
            REPO / "dram" / "out" / "litedram_core.v": src / "litedram_core.v",
            REPO / "hw_build" / cstdir / "pins.cst": src / "pins.cst",
        }
        for a, b in files.items():
            if not a.exists():
                raise SystemExit(f"missing artifact: {a} (compile first?)")
            shutil.copy2(a, b)
            print(f"[stage] {a.name} ({a.stat().st_size // 1024} KiB)")

    # 3. build.tcl + gw_sh -------------------------------------------------
    if not args.report_only:
        tcl = "\n".join([
            f"set_device -device_version {devver} {part}",
            'add_file "src/DramSoCTop.v"',
            'add_file "src/litedram_core.v"',
            'add_file "src/pins.cst"',
            f"set_option -top_module {args.top}",
            "set_option -output_base_name DRAM",
            "run all",
            "exit",
            "",
        ])
        (work / "build.tcl").write_text(tcl, encoding="utf-8")
        if not GOWIN_SH.exists():
            raise SystemExit(f"gw_sh not found: {GOWIN_SH}")
        sh([GOWIN_SH, "build.tcl"], work, "gw_sh")

    # 4. report -------------------------------------------------------------
    impl = work / "impl"
    rsc = impl / "gwsynthesis" / "DRAM_syn_rsc.xml"
    rpt = impl / "pnr" / "DRAM.rpt.txt"
    fs = impl / "pnr" / "DRAM.fs"
    if not (rsc.exists() and rpt.exists() and fs.exists()):
        raise SystemExit("[report] missing outputs "
                         f"(rsc={rsc.exists()} rpt={rpt.exists()} fs={fs.exists()})")
    m = re.search(r'T_Register="(\d+).*?T_Lut="(\d+)', rsc.read_text(),
                  re.DOTALL)
    lut = int(m.group(2)) if m else -1
    reg = int(m.group(1)) if m else -1
    bsram = re.search(r'T_Bsram="(\d+)', rsc.read_text())
    bsram = int(bsram.group(1)) if bsram else -1
    txt = rpt.read_text(errors="replace")
    errs = [l for l in txt.splitlines() if "ERROR" in l]
    dll = re.search(r"DLL\s+\|\s+(\d+)", txt)
    dqs = re.search(r"DQS\s+\|\s+(\d+)", txt)
    print("=" * 60)
    print(f"[report] LUT={lut} REG={reg} BSRAM={bsram} "
          f"DLL={dll.group(1) if dll else '?'} DQS={dqs.group(1) if dqs else '?'} "
          f"ERRORs={len(errs)} fs={fs.stat().st_size // 1024} KiB")
    ok = (EXPECTED["lut_min"] <= lut <= EXPECTED["lut_max"]
          and EXPECTED["reg_min"] <= reg <= EXPECTED["reg_max"]
          and 0 <= bsram <= EXPECTED["bsram_max"]
          and not errs and fs.stat().st_size > 1_000_000)
    print("[report] " + ("PASS" if ok else "CHECK MANUALLY"))
    if not ok:
        raise SystemExit(1)

    # 5. flash (optional) ----------------------------------------------------
    if args.flash == "programmer":
        sh([PROGRAMMER_CLI, "--device", "GW2A-18C", "--fsFile", str(fs)],
           REPO, "flash")
    elif args.flash == "openfpgaloader":
        sh([sys.executable, "cli/main.py", "flash", str(fs)], REPO, "flash")

    # 6. validate (optional) --------------------------------------------------
    if args.validate:
        sh([sys.executable, r"examples\UniversalScale\silicon_validate.py"],
           REPO, "validate")


if __name__ == "__main__":
    main()
