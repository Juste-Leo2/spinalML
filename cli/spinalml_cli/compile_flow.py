# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""`spinalml compile` flow (moved verbatim from cli.py:compile).

Decomposition: unwrap_options -> resolve_target -> prepare_workspace ->
detect_entrypoint -> build_autorunner -> run_elaboration ->
collect_artifacts (+ optional run_chain_step) -> print_summary.
`run_tool` is injected by cli.py (it lives there and is shared by all
tool-passthrough commands; importing it here would cycle cli -> compile_flow).
"""

import glob
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import typer
from typer.models import ArgumentInfo, OptionInfo

from .board import detect_model_parameters, load_board_config, parse_frequency
from .config import get_active_framework_root, get_project_root


def _unwrap(val, default=None):
    if isinstance(val, (OptionInfo, ArgumentInfo)):
        return val.default if val.default is not ... else default
    return val


def resolve_rounding(rounding: Optional[str]) -> Tuple[Optional[str], str]:
    """Resolve the elaboration rounding flag.

    Returns (SPINALML_ROUNDING value to set or None to keep the pre-set env,
    display label). None keeps the environment untouched; RoundingConfig
    itself falls back to RNE when neither flag nor env is set.
    """
    trunc_values = ("trunc", "truncate", "floor")
    is_trunc = str(rounding).strip().lower() in trunc_values if rounding is not None \
        else os.environ.get("SPINALML_ROUNDING", "").strip().lower() in trunc_values
    label = "Truncate (legacy bit-exact)" if is_trunc else "RNE (default, unbiased)"
    if rounding is None:
        return None, label
    return ("trunc" if is_trunc else "rne"), label


def unwrap_options(file, out, chain, soc, board, clk, baud, out_count,
                   word_width, bram_words, no_dsp, rounding) -> dict:
    """Unwraps typer options (supports programmatic OptionInfo values) + file/out setup."""
    opts = {
        "file": _unwrap(file),
        "out": _unwrap(out, Path("rtl")),
        "chain": _unwrap(chain, True),
        "soc": _unwrap(soc, False),
        "board": _unwrap(board, "tang-primer-20k"),
        "clk": _unwrap(clk, None),
        "baud": _unwrap(baud, None),
        "out_count": _unwrap(out_count, None),
        "word_width": _unwrap(word_width, None),
        "bram_words": _unwrap(bram_words, None),
        "no_dsp": _unwrap(no_dsp, False),
        "rounding": _unwrap(rounding, None),
    }
    if not opts["file"].exists():
        typer.echo(f"Error: File {file} does not exist.", err=True)
        raise typer.Exit(code=1)

    opts["out"].mkdir(parents=True, exist_ok=True)
    opts["target_dir"] = str(opts["out"].resolve()).replace('\\', '/')
    return opts


def resolve_target(opts: dict) -> dict:
    """Loads board config, sets target env, introspects the model, echoes the plan."""
    board = opts["board"]
    try:
        board_cfg = load_board_config(board)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Configure DSP target environment for SpinalHDL compilation
    vendor = board_cfg.get("vendor", "Generic")
    os.environ["SPINALML_TARGET"] = vendor
    if opts["no_dsp"]:
        os.environ["SPINALML_NO_DSP"] = "1"
    elif "SPINALML_NO_DSP" in os.environ:
        del os.environ["SPINALML_NO_DSP"]

    # Rounding policy for elaboration (mirrors RoundingConfig.current):
    # explicit flag wins, otherwise a pre-set SPINALML_ROUNDING is kept
    # (RoundingConfig itself falls back to RNE when unset).
    rounding_value, rounding_label = resolve_rounding(opts["rounding"])
    if rounding_value is not None:
        os.environ["SPINALML_ROUNDING"] = rounding_value

    model_params = detect_model_parameters(opts["file"])
    final_clk = parse_frequency(opts["clk"]) if opts["clk"] else board_cfg["clk_freq"]
    final_baud = opts["baud"] if opts["baud"] is not None else board_cfg["baud_rate"]
    final_out_count = opts["out_count"] if opts["out_count"] is not None else model_params.get("out_count", 10)
    final_word_width = opts["word_width"] if opts["word_width"] is not None else model_params.get("word_width", 64)
    final_bram_words = opts["bram_words"] if opts["bram_words"] is not None else board_cfg.get("bram_words", 4096)

    dsp_status = "Disabled (LUTs only)" if opts["no_dsp"] else f"Enabled ({vendor} DSP mapping)"
    typer.echo(f"Target Board   : {board_cfg['name']} ({board_cfg.get('fpga', 'FPGA')})")
    typer.echo(f"DSP Policy     : {dsp_status}")
    typer.echo(f"Rounding       : {rounding_label}")
    typer.echo(f"Hardware Clock : {final_clk / 1e6:.2f} MHz | UART: {final_baud} baud (CLK_PER_BIT = {final_clk // final_baud})")
    typer.echo(f"Model Protocol : {final_out_count} output bytes | {final_word_width}-bit AXI | {final_bram_words} words BRAM")

    return {
        "board_cfg": board_cfg,
        "final_clk": final_clk,
        "final_baud": final_baud,
        "final_out_count": final_out_count,
        "final_word_width": final_word_width,
        "final_bram_words": final_bram_words,
    }


def prepare_workspace(opts: dict) -> Tuple[Path, bool]:
    """Copies external files into the mill workspace. Returns (workspace_src, is_internal)."""
    framework_root = get_active_framework_root()
    workspace_src = framework_root / "spinalML" / "src" / "cli_temp"
    if workspace_src.exists():
        shutil.rmtree(workspace_src)
    workspace_src.mkdir(parents=True, exist_ok=True)

    # Check if file is already in spinalML/src
    file = opts["file"]
    spinalml_src = framework_root / "spinalML" / "src"
    try:
        is_internal = file.resolve().is_relative_to(spinalml_src.resolve())
    except AttributeError:
        is_internal = str(file.resolve()).startswith(str(spinalml_src.resolve()))

    if not is_internal:
        shutil.copy(file, workspace_src / file.name)
        typer.echo(f"Copied external file {file.name} to temporary workspace.")
    return workspace_src, is_internal


def detect_entrypoint(opts: dict) -> dict:
    """Detects the App entrypoint or Component/Accelerator class in the file."""
    content = opts["file"].read_text(encoding="utf-8")
    pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
    pkg = pkg_match.group(1) if pkg_match else ""
    app_match = re.search(r'^\s*object\s+(\w+)\s+extends\s+App', content, re.MULTILINE)

    comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
    is_accelerator = bool(comp_match and (("extends Accelerator" in content) or ("extends Accelerator" in comp_match.group(0))))
    return {"content": content, "pkg": pkg, "app_match": app_match,
            "comp_match": comp_match, "is_accelerator": is_accelerator}


def build_autorunner(opts: dict, tgt: dict, entry: dict, workspace_src: Path) -> Tuple[str, bool]:
    """Writes AutoRunner.scala when needed. Returns (full_main, auto_generated)."""
    target_dir = opts["target_dir"]
    final_clk, final_baud = tgt["final_clk"], tgt["final_baud"]
    final_out_count, final_word_width, final_bram_words = tgt["final_out_count"], tgt["final_word_width"], tgt["final_bram_words"]

    # If --soc is requested and an Accelerator is present, or if no App entrypoint exists, use AutoRunner
    use_autorunner = (opts["soc"] and entry["is_accelerator"]) or not entry["app_match"]

    if not use_autorunner and entry["app_match"]:
        main_class = entry["app_match"].group(1)
        return (f"{entry['pkg']}.{main_class}" if entry["pkg"] else main_class), False

    if not entry["comp_match"]:
        typer.echo(f"Error: {opts['file'].name} does not contain 'object <Name> extends App' nor a Component.", err=True)
        typer.echo("Please add an App entry point to generate Verilog.", err=True)
        shutil.rmtree(workspace_src)
        raise typer.Exit(code=1)

    comp_name = entry["comp_match"].group(1)
    import_stmt = f"import {entry['pkg']}.{comp_name}" if entry["pkg"] else ""

    soc_snippet = ""
    if opts["soc"] and entry["is_accelerator"]:
        soc_snippet = f"""
  println(s"[AutoRunner] Generating complete turnkey UartSoC top-level in '{target_dir}'...")
  val cfg = spinal.lib.bus.amba4.axi.Axi4Config(addressWidth = 32, dataWidth = {final_word_width}, idWidth = 4)
  spinalConfig.generateVerilog(new spinalML.io.UartSoC(
    acceleratorFactory = () => new {comp_name}(),
    clkFreq = BigInt({final_clk}),
    baudRate = BigInt({final_baud}),
    axiConfig = cfg,
    memoryWords = {final_bram_words},
    outCount = {final_out_count}
  ))
"""
    elif opts["soc"] and not entry["is_accelerator"]:
        typer.echo(f"[Notice] --soc requested, but {comp_name} does not extend Accelerator. Skipping UartSoC top-level.")

    chain_snippet = ""
    if opts["chain"]:
        chain_snippet = f"""
  println(s"[AutoRunner] Generating supplementary UART chain Verilog in '{target_dir}'...")
  val chainCfg = spinal.lib.bus.amba4.axi.Axi4Config(addressWidth = 32, dataWidth = {final_word_width}, idWidth = 4)
  spinalConfig.generateVerilog(new spinalML.io.UartRx(BigInt("{final_clk}"), BigInt("{final_baud}")))
  spinalConfig.generateVerilog(new spinalML.io.UartTx(BigInt("{final_clk}"), BigInt("{final_baud}")))
  spinalConfig.generateVerilog(new spinalML.io.UartBridge(outCount = {final_out_count}, wordWidth = chainCfg.dataWidth, csrAddrWidth = 8, version = 0x01))
  spinalConfig.generateVerilog(new spinalML.io.AxiReadMem(chainCfg, memoryWords = {final_bram_words}, imgBase = 0x10000, weightBase = 0x20000))
"""

    auto_runner_code = f"""
package spinalml_auto
import spinal.core._
{import_stmt}

object AutoRunner extends App {{
  val spinalConfig = SpinalConfig(
    targetDirectory = "{target_dir}",
    headerWithDate = true,
    rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
  )
  spinalConfig.generateVerilog(new {comp_name}())
{soc_snippet}
{chain_snippet}
}}
"""
    (workspace_src / "AutoRunner.scala").write_text(auto_runner_code, encoding="utf-8")
    typer.echo(f"Auto-generating runner for component {comp_name}...")
    return "spinalml_auto.AutoRunner", True


def run_elaboration(opts: dict, full_main: str, auto_generated: bool,
                    workspace_src: Path, run_tool: Callable) -> None:
    """Runs Mill elaboration, cleans the workspace, raises Exit on failure."""
    typer.echo(f"Running Mill spinalML.runMain {full_main}...")
    ret_code = run_tool("mill", ["--no-server", "spinalML.runMain", full_main], exit_on_error=False)

    if workspace_src.exists():
        shutil.rmtree(workspace_src)

    if ret_code != 0:
        if auto_generated:
            typer.echo("\n" + "=" * 60, err=True)
            typer.echo("Failed to auto-instantiate the component.", err=True)
            typer.echo("If your component requires mandatory arguments (like Axi4Config),", err=True)
            typer.echo("please add an `object YourGenerator extends App` block in your file.", err=True)
            typer.echo("=" * 60 + "\n", err=True)
        raise typer.Exit(code=ret_code)


def collect_artifacts(project_root: Path, out: Path, start_time: float) -> None:
    """Moves new root-level .v/.bin artifacts into the output directory."""
    for f_path in (glob.glob(str(project_root / "*.v")) + glob.glob(str(project_root / "*.bin"))):
        p = Path(f_path)
        if p.name in ["top.v", "uart_rx.v", "uart_tx.v"]:
            continue
        try:
            if os.path.getmtime(f_path) >= start_time:
                dest = out / p.name
                if dest.resolve() != p.resolve():
                    shutil.move(f_path, dest)
                    typer.echo(f"Saved {p.name} -> {out}")
        except OSError:
            pass


def run_chain_step(opts: dict, tgt: dict, auto_generated: bool, run_tool: Callable) -> None:
    """Generates supplementary UART chain files (non-AutoRunner path only)."""
    if not (opts["chain"] and not auto_generated):
        return
    typer.echo(f"\nGenerating supplementary UART chain Verilog in {opts['out']}...")
    chain_args = [
        "--no-server",
        "spinalML.runMain", "spinalML.io.UartChainGen",
        "--out", opts["target_dir"],
        "--clk", str(tgt["final_clk"]),
        "--baud", str(tgt["final_baud"]),
        "--out-count", str(tgt["final_out_count"]),
        "--word-width", str(tgt["final_word_width"]),
        "--memory-words", str(tgt["final_bram_words"])
    ]
    run_tool("mill", chain_args, exit_on_error=False)


def print_summary(out: Path) -> None:
    """Lists generated hardware files."""
    generated_v = sorted(out.glob("*.v"))
    generated_bin = sorted(out.glob("*.bin"))
    typer.echo(f"\nCompilation complete. Hardware files available in '{out}':")
    for v in generated_v:
        typer.echo(f"  [Verilog] {v.name}")
    for b in generated_bin:
        typer.echo(f"  [Memory]  {b.name}")


def run_compile(file: Path,
                out: Path,
                chain: bool,
                soc: bool,
                board: str,
                clk: Optional[str],
                baud: Optional[int],
                out_count: Optional[int],
                word_width: Optional[int],
                bram_words: Optional[int],
                no_dsp: bool,
                rounding: Optional[str],
                run_tool: Callable[[str, List[str]], int]) -> None:
    """Full `spinalml compile` flow (raises typer.Exit on failure)."""
    opts = unwrap_options(file, out, chain, soc, board, clk, baud, out_count,
                          word_width, bram_words, no_dsp, rounding)
    tgt = resolve_target(opts)
    workspace_src, _ = prepare_workspace(opts)
    entry = detect_entrypoint(opts)
    full_main, auto_generated = build_autorunner(opts, tgt, entry, workspace_src)

    project_root = get_project_root()
    start_time = time.time() - 2

    run_elaboration(opts, full_main, auto_generated, workspace_src, run_tool)
    collect_artifacts(project_root, opts["out"], start_time)
    run_chain_step(opts, tgt, auto_generated, run_tool)
    print_summary(opts["out"])
