import re
import subprocess
import sys
import time
import typer
from typing import List, Optional
from pathlib import Path

from .config import load_config, get_bin_path, CLI_DIR
from .installer import setup_tools
from .board import load_board_config, parse_frequency, detect_model_parameters, list_available_boards

app = typer.Typer(
    help="SpinalML CLI - Wrapper for FPGA Tools",
    context_settings={"help_option_names": ["-h", "--help"]}
)

@app.command()
def setup(
    debug: bool = typer.Option(False, "--debug", help="Show verbose raw logs"),
    force: bool = typer.Option(False, "-f", "--force", help="Force reinstallation of tools even if already up to date"),
    clean_cache: bool = typer.Option(False, "--clean-cache", help="Clean Coursier & Ivy caches before setup")
):
    """
    Download and extract all necessary tools (Mill, OSS CAD Suite) to ~/.spinalml_tools
    """
    config = load_config()
    setup_tools(config, debug=debug, force=force, clean_cache=clean_cache)

@app.command(name="clean-cache")
def clean_cache(
    debug: bool = typer.Option(False, "--debug", help="Show verbose raw logs")
):
    """
    Remove Coursier and Ivy caches to prevent checksum corruption.
    """
    from .installer import clean_coursier_cache
    from rich.console import Console
    console = Console()
    clean_coursier_cache(console=console, debug=debug)
    console.print("[bold green]Coursier & Ivy caches cleaned successfully![/bold green]")

def run_tool(tool_name: str, args: List[str], exit_on_error: bool = True) -> int:
    """Helper to run an installed tool and pass along arguments."""
    bin_path = get_bin_path(tool_name)
    
    if not bin_path.exists():
        typer.echo(f"Error: {tool_name} is not installed at {bin_path}.", err=True)
        typer.echo("Please run 'spinalml setup' first.", err=True)
        if exit_on_error:
            raise typer.Exit(code=1)
        return 1
    
    # Build command
    cmd = [str(bin_path)] + args
    
    # Temporarily set PATH and environment variables so tools can find dependencies
    import os
    
    from .config import TOOLS_DIR
    oss_bin = TOOLS_DIR / "oss-cad-suite" / "bin"
    oss_lib = TOOLS_DIR / "oss-cad-suite" / "lib"
    w64_bin = TOOLS_DIR / "w64devkit" / "bin"
    verilator_root = TOOLS_DIR / "oss-cad-suite" / "share" / "verilator"
    
    new_paths = [str(oss_bin), str(oss_lib)]
    if verilator_root.exists():
        os.environ["VERILATOR_ROOT"] = str(verilator_root)
        verilator_bin = verilator_root / "bin"
        if verilator_bin.exists():
            new_paths.append(str(verilator_bin))
            
    if w64_bin.exists():
        new_paths.append(str(w64_bin))
    if str(bin_path.parent) not in new_paths:
        new_paths.insert(0, str(bin_path.parent))
        
    existing_path = os.environ.get("PATH", "") or os.environ.get("Path", "")
    combined_path = os.pathsep.join(new_paths) + os.pathsep + existing_path
    os.environ["PATH"] = combined_path
    os.environ["Path"] = combined_path
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=str(CLI_DIR.parent))
        if exit_on_error and result.returncode != 0:
            sys.exit(result.returncode)
        return result.returncode
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        typer.echo(f"Error executing {tool_name}: {e}", err=True)
        if exit_on_error:
            sys.exit(1)
        return 1

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def mill(ctx: typer.Context):
    """
    Run Mill build tool
    """
    run_tool("mill", ctx.args)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def verilator(ctx: typer.Context):
    """
    Run Verilator
    """
    run_tool("verilator", ctx.args)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sby(ctx: typer.Context):
    """
    Run SymbiYosys (sby)
    """
    run_tool("sby", ctx.args)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def nextpnr(ctx: typer.Context, arch: str = typer.Argument(..., help="Architecture (e.g. ice40, ecp5, himbaechel)")):
    """
    Run nextpnr. Use args to specify architecture (e.g. 'spinalml nextpnr ice40')
    """
    run_tool(f"nextpnr-{arch}", ctx.args)
    
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def yosys(ctx: typer.Context):
    """
    Run Yosys
    """
    run_tool("yosys", ctx.args)

@app.command(name="openfpgaloader", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def open_fpga_loader(ctx: typer.Context):
    """
    Run openFPGALoader
    """
    run_tool("openFPGALoader", ctx.args)

@app.command()
def compile(
    file: Path = typer.Argument(..., help="Path to the Scala file to compile"),
    out: Path = typer.Option(Path("rtl"), "-o", "--out", help="Output directory for generated Verilog files [default: rtl]"),
    chain: bool = typer.Option(True, "--chain/--no-chain", help="Generate supplementary UART chain Verilog files (UartRx, UartTx, UartBridge, AxiReadMem)"),
    soc: bool = typer.Option(False, "--soc/--no-soc", help="Generate complete turnkey UartSoC top-level wrapping the accelerator"),
    board: str = typer.Option("tang-primer-20k", "--board", help="Target FPGA board profile from boards/*.json [default: tang-primer-20k]"),
    clk: Optional[str] = typer.Option(None, "--clk", help="Clock frequency override (e.g. '27MHz', '50MHz', '100MHz')"),
    baud: Optional[int] = typer.Option(None, "--baud", help="UART baudrate override (default: from board or 115200)"),
    out_count: Optional[int] = typer.Option(None, "--out-count", help="Number of output stream bytes/logits (auto-detected from model if omitted)"),
    word_width: Optional[int] = typer.Option(None, "--word-width", help="AXI data bus width in bits (auto-detected from model if omitted)"),
    bram_words: Optional[int] = typer.Option(None, "--bram-words", help="BRAM capacity in 64-bit words (default: from board or 4096)"),
):
    """
    Compile a Scala file into Verilog by running it within the workspace module,
    and generate the supplementary hardware chain files (UartRx, UartTx, UartBridge,
    AxiReadMem) into the output directory with FPGA board-specific settings.
    """
    import shutil
    import os
    import glob
    import re
    from typer.models import OptionInfo, ArgumentInfo

    def _unwrap(val, default=None):
        if isinstance(val, (OptionInfo, ArgumentInfo)):
            return val.default if val.default is not ... else default
        return val

    out = _unwrap(out, Path("rtl"))
    chain = _unwrap(chain, True)
    soc = _unwrap(soc, False)
    board = _unwrap(board, "tang-primer-20k")
    clk = _unwrap(clk, None)
    baud = _unwrap(baud, None)
    out_count = _unwrap(out_count, None)
    word_width = _unwrap(word_width, None)
    bram_words = _unwrap(bram_words, None)

    if not file.exists():
        typer.echo(f"Error: File {file} does not exist.", err=True)
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    target_dir = str(out.resolve()).replace('\\', '/')

    # 1. Resolve board configuration & model introspection
    try:
        board_cfg = load_board_config(board)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    model_params = detect_model_parameters(file)
    final_clk = parse_frequency(clk) if clk else board_cfg["clk_freq"]
    final_baud = baud if baud is not None else board_cfg["baud_rate"]
    final_out_count = out_count if out_count is not None else model_params.get("out_count", 10)
    final_word_width = word_width if word_width is not None else model_params.get("word_width", 64)
    final_bram_words = bram_words if bram_words is not None else board_cfg.get("bram_words", 4096)

    typer.echo(f"Target Board   : {board_cfg['name']} ({board_cfg.get('fpga', 'FPGA')})")
    typer.echo(f"Hardware Clock : {final_clk/1e6:.2f} MHz | UART: {final_baud} baud (CLK_PER_BIT = {final_clk // final_baud})")
    typer.echo(f"Model Protocol : {final_out_count} output bytes | {final_word_width}-bit AXI | {final_bram_words} words BRAM")

    content = file.read_text(encoding="utf-8")
    pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
    pkg = pkg_match.group(1) if pkg_match else ""
    app_match = re.search(r'^\s*object\s+(\w+)\s+extends\s+App', content, re.MULTILINE)

    workspace_src = CLI_DIR.parent / "spinalML" / "src" / "cli_temp"
    if workspace_src.exists():
        shutil.rmtree(workspace_src)
    workspace_src.mkdir(parents=True, exist_ok=True)

    # Check if file is already in spinalML/src
    spinalml_src = CLI_DIR.parent / "spinalML" / "src"
    try:
        is_internal = file.resolve().is_relative_to(spinalml_src.resolve())
    except AttributeError:
        is_internal = str(file.resolve()).startswith(str(spinalml_src.resolve()))

    if not is_internal:
        shutil.copy(file, workspace_src / file.name)
        typer.echo(f"Copied external file {file.name} to temporary workspace.")

    full_main = ""
    auto_generated = False

    comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
    is_accelerator = bool(comp_match and (("extends Accelerator" in content) or ("extends Accelerator" in comp_match.group(0))))

    # If --soc is requested and an Accelerator is present, or if no App entrypoint exists, use AutoRunner
    use_autorunner = (soc and is_accelerator) or not app_match

    if not use_autorunner and app_match:
        main_class = app_match.group(1)
        full_main = f"{pkg}.{main_class}" if pkg else main_class
    else:
        if not comp_match:
            typer.echo(f"Error: {file.name} does not contain 'object <Name> extends App' nor a Component.", err=True)
            typer.echo("Please add an App entry point to generate Verilog.", err=True)
            shutil.rmtree(workspace_src)
            raise typer.Exit(code=1)

        comp_name = comp_match.group(1)
        import_stmt = f"import {pkg}.{comp_name}" if pkg else ""

        soc_snippet = ""
        if soc and is_accelerator:
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
        elif soc and not is_accelerator:
            typer.echo(f"[Notice] --soc requested, but {comp_name} does not extend Accelerator. Skipping UartSoC top-level.")

        chain_snippet = ""
        if chain:
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
        full_main = "spinalml_auto.AutoRunner"
        auto_generated = True
        typer.echo(f"Auto-generating runner for component {comp_name}...")

    project_root = CLI_DIR.parent
    start_time = time.time() - 2

    typer.echo(f"Running Mill spinalML.runMain {full_main}...")
    ret_code = run_tool("mill", ["--no-server", "spinalML.runMain", full_main], exit_on_error=False)

    if workspace_src.exists():
        shutil.rmtree(workspace_src)

    if ret_code != 0:
        if auto_generated:
            typer.echo("\n" + "="*60, err=True)
            typer.echo("Failed to auto-instantiate the component.", err=True)
            typer.echo("If your component requires mandatory arguments (like Axi4Config),", err=True)
            typer.echo("please add an `object YourGenerator extends App` block in your file.", err=True)
            typer.echo("="*60 + "\n", err=True)
        raise typer.Exit(code=ret_code)

    def move_new_root_artifacts():
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

    # Move any new .v or .bin files generated at root to destination
    move_new_root_artifacts()

    # Generate supplementary UART chain files if requested and not already in AutoRunner
    if chain and not auto_generated:
        typer.echo(f"\nGenerating supplementary UART chain Verilog in {out}...")
        chain_args = [
            "--no-server",
            "spinalML.runMain", "spinalML.io.UartChainGen",
            "--out", target_dir,
            "--clk", str(final_clk),
            "--baud", str(final_baud),
            "--out-count", str(final_out_count),
            "--word-width", str(final_word_width),
            "--memory-words", str(final_bram_words)
        ]
        run_tool("mill", chain_args, exit_on_error=False)
        move_new_root_artifacts()

    generated_v = sorted(out.glob("*.v"))
    generated_bin = sorted(out.glob("*.bin"))
    typer.echo(f"\nCompilation complete. Hardware files available in '{out}':")
    for v in generated_v:
        typer.echo(f"  [Verilog] {v.name}")
    for b in generated_bin:
        typer.echo(f"  [Memory]  {b.name}")


def _run_single_test_file(target_file: Path) -> int:
    import re
    import shutil

    content = target_file.read_text(encoding="utf-8")
    pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
    pkg = pkg_match.group(1) if pkg_match else ""

    # 1. Check if the file is an existing ScalaTest suite
    test_suite_match = re.search(r'class\s+(\w+)\s+extends\s+AnyFunSuite', content)
    if test_suite_match:
        test_class = test_suite_match.group(1)
        full_test = f"{pkg}.{test_class}" if pkg else test_class
        typer.echo(f"Detected ScalaTest suite: {full_test}")
        typer.echo(f"Running Mill testOnly {full_test}...")
        ret_code = run_tool("mill", ["--no-server", "spinalML.test.testOnly", full_test], exit_on_error=False)
        if ret_code == 0:
            typer.echo("All tests passed successfully!")
        return ret_code

    # 2. Check for component/accelerator
    comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
    if comp_match:
        comp_name = comp_match.group(1)
        
        # Run the Universal Bit-Exact Verification Engine using UniversalTestHarness
        test_temp_dir = CLI_DIR.parent / "spinalML" / "test" / "src" / "cli_test_temp"
        if test_temp_dir.exists():
            shutil.rmtree(test_temp_dir)
        test_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # If the file is located outside the spinalML source tree (e.g. at repo root or tests/universal),
            # copy it into test_temp_dir so Mill automatically compiles it alongside the test scaffold.
            try:
                target_file.resolve().relative_to((CLI_DIR.parent / "spinalML" / "src").resolve())
            except ValueError:
                try:
                    target_file.resolve().relative_to((CLI_DIR.parent / "spinalML" / "test" / "src").resolve())
                except ValueError:
                    shutil.copy(target_file, test_temp_dir / target_file.name)

            import_stmt = f"import {pkg}.{comp_name}" if pkg else f"import _root_.{comp_name}"
            scaffold_code = f"""package cli_test_temp

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.harness.{{MemoryHarness, UniversalTestHarness}}
import spinalML.replica.{{HWArithmetic, WeightMemoryLayout, ModelReplica}}
{import_stmt}

class AutoGeneratedCircuitTest extends AnyFunSuite {{
  test("{comp_name} universal bit-exact circuit verification under Verilator") {{
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    
    var replicaLogits: Seq[Double] = null
    var packedWords: Seq[BigInt] = null
    var imgWords: Seq[BigInt] = null

    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile {{
      val dut = new {comp_name}()
      val layers = dut.modelSpec
      val inputShape = dut.inputShape
      val pipelineDtype = dut.globalDataType
      
      // 1. Build deterministic test weights packed to AXI beats
      val packed = WeightMemoryLayout.buildDeterministicWeights(layers, pipelineDtype, axiConfig)
      packedWords = packed.words
      
      // 2. Build deterministic input tensor
      val inElems = inputShape.product
      val inData = pipelineDtype()
      val isIntInput = inData.isInstanceOf[SInt] || inData.isInstanceOf[UInt]
      val inputTensor = if (isIntInput) {{
        val inInts = (0 until inElems).map {{ idx =>
          if (inData.isInstanceOf[UInt]) (((idx * 7 + 3) % 15)).toLong
          else (((idx * 7 + 3) % 15) - 7).toLong
        }}
        imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
        ModelReplica.IntTensor(inputShape, inInts, inData.getBitsWidth)
      }} else {{
        val (eW, mW) = inData match {{
          case f: spinalML.dtypes.FloatML => (f.expBits, f.mantBits)
          case _ => (8, 7)
        }}
        val inputValues = (0 until inElems).map {{ idx =>
          val sign = if (idx % 2 == 0) 1.0f else -1.0f
          val mag = (((idx % 7) + 1) * 0.125).toFloat
          HWArithmetic.fromDouble(sign * mag, eW, mW)
        }}
        val elemBits = 1 + eW + mW
        val rawBits = inputValues.map {{ f =>
          val sign = if (f.s) 1L else 0L
          (sign << (eW + mW)) | ((f.e.toLong & ((1L << eW) - 1)) << mW) | (f.m.toLong & ((1L << mW) - 1))
        }}
        imgWords = if (elemBits <= 8) {{
          MemoryHarness.packBytes(rawBits.map(_.toInt))
        }} else {{
          val inFloats = inputValues.map(f => HWArithmetic.decode(f, eW, mW).toFloat)
          MemoryHarness.packFloats(MemoryHarness.padded(inFloats))
        }}
        ModelReplica.FloatTensor(inputShape, inputValues, eW, mW)
      }}
      
      // 3. Compute software oracle with intermediate activation previews
      val replicaResult = ModelReplica.forwardWithTrace(layers, inputShape, inputTensor, packed)
      replicaLogits = replicaResult.logits
      dut
    }}
    
    val output = UniversalTestHarness.run(
      compiled = compiled,
      weightsWords = packedWords,
      imageWords = imgWords,
      expectedLogits = Some(replicaLogits)
    )
    
    println(s"[{comp_name}] Collected ${{output.length}} outputs. Bit-exact verification complete!")
  }}
}}
"""
            (test_temp_dir / "AutoGeneratedCircuitTest.scala").write_text(scaffold_code, encoding="utf-8")
            full_test = "cli_test_temp.AutoGeneratedCircuitTest"

            typer.echo(f"Scaffolded simulation test for {comp_name}...")
            typer.echo(f"Running Mill testOnly {full_test}...")

            ret_code = run_tool("mill", ["--no-server", "spinalML.test.testOnly", full_test], exit_on_error=False)
            if ret_code == 0:
                typer.echo(f"Circuit verification for {comp_name} passed successfully!")
            return ret_code
        finally:
            if test_temp_dir.exists():
                shutil.rmtree(test_temp_dir)

    # 3. Check if the file is an executable test object (main or extends App)
    app_test_match = re.search(r'object\s+(\w+)(?:\s+extends\s+App|\s*\{[^}]*def\s+main)', content, re.DOTALL)
    if app_test_match:
        test_obj = app_test_match.group(1)
        full_test = f"{pkg}.{test_obj}" if pkg else test_obj
        typer.echo(f"Detected executable test object: {full_test}")
        typer.echo(f"Running Mill test.runMain {full_test}...")
        ret_code = run_tool("mill", ["--no-server", "spinalML.test.runMain", full_test], exit_on_error=False)
        if ret_code == 0:
            typer.echo(f"Test {test_obj} passed successfully!")
        return ret_code

    typer.echo(f"Error: Could not identify a test suite or Component/Accelerator in {target_file.name}.", err=True)
    return 1


@app.command()
def test(
    file: Path = typer.Argument(..., help="Path to the Scala model or test file/directory to run"),
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test sequences (0 = disabled; use e.g. --ci 3 on slow-SD/self-hosted runners)")
):
    """
    Run hardware simulation tests (bit-exact, streaming, tiling, memory verification).
    Supports single Scala files or directories containing test circuits.
    """
    if not file.exists():
        typer.echo(f"Error: Path {file} does not exist.", err=True)
        raise typer.Exit(code=1)

    if file.is_dir():
        scala_files = sorted(file.glob("*.scala"))
        if not scala_files:
            typer.echo(f"Error: No .scala test files found in {file}.", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Found {len(scala_files)} test suite(s) in {file}: {[f.name for f in scala_files]}")
        failed = []
        for idx, f in enumerate(scala_files, 1):
            typer.echo(f"\n========================================================")
            typer.echo(f"[{idx}/{len(scala_files)}] Running: {f.name}")
            typer.echo(f"========================================================")
            rc = _run_single_test_file(f)
            if rc != 0:
                failed.append(f.name)
            if ci_sleep > 0 and idx < len(scala_files):
                time.sleep(ci_sleep)

        typer.echo(f"\n========================================================")
        if failed:
            typer.echo(f"FAILED: {len(failed)}/{len(scala_files)} test(s) failed: {', '.join(failed)}", err=True)
            raise typer.Exit(code=1)
        else:
            typer.echo(f"SUCCESS: All {len(scala_files)} test(s) passed successfully!")
            return

    rc = _run_single_test_file(file)
    if rc != 0:
        raise typer.Exit(code=rc)

@app.command(name="test-all")
def test_all(
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter tests by name pattern (regex or substring)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print failure logs and error traces directly to the terminal"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory to store failure logs (default: out/test_reports)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List discovered test suites without running them"),
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test sequences (0 = disabled; use e.g. --ci 3 on slow-SD/self-hosted runners)"),
):
    """
    Run all ScalaTest suites sequentially (1-by-1) to avoid Verilator/G++ RAM exhaustion.
    Captures failure traces into individual log files under out/test_reports/.
    """
    from .test_runner import run_all_tests
    code = run_all_tests(filter_pattern=filter, fail_fast=fail_fast, log_dir=log_dir, dry_run=dry_run, verbose=verbose, ci_sleep=ci_sleep)
    if code != 0:
        raise typer.Exit(code=code)

@app.command(name="test-all-formal")
def test_all_formal(
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter formal tests by name pattern (regex or substring)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print failure logs and error traces directly to the terminal"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory to store failure logs (default: out/formal_reports)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List discovered formal suites without running them"),
    timeout: int = typer.Option(900, "-t", "--timeout", help="Timeout per formal suite in seconds (default: 900)"),
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test sequences (0 = disabled; use e.g. --ci 3 on slow-SD/self-hosted runners)"),
):
    """
    Run all SymbiYosys formal verification suites sequentially (1-by-1).
    Discovers all *Formal.scala specs under symbolicTest/ and executes them via SMT-BMC.
    """
    from .formal_runner import run_all_formal_tests
    code = run_all_formal_tests(filter_pattern=filter, fail_fast=fail_fast, log_dir=log_dir, dry_run=dry_run, timeout=timeout, verbose=verbose, ci_sleep=ci_sleep)
    if code != 0:
        raise typer.Exit(code=code)

@app.command(name="test-all-python", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test_all_python(
    ctx: typer.Context,
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter tests by name pattern (regex or substring)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print failure logs and error traces directly to the terminal"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory to store failure logs (default: out/python_reports)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List discovered Python test files without running them"),
    debug_math: bool = typer.Option(False, "--debug-math", help="Generate true_math_errors.log with detailed precision errors"),
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test file sequences (0 = disabled; use e.g. --ci 3 on slow-SD/self-hosted runners)"),
):
    """
    Run Python Cocotb/Pytest hardware co-simulations sequentially (file-by-file) with Verilator.
    Requires Linux (or Windows via WSL) due to Cocotb VPI bridge architecture.
    """
    if sys.platform == "win32":
        typer.secho(
            "\n[Notice] Cocotb + Verilator hardware co-simulations require a Linux environment (Linux, Radxa ARM64, or Windows via WSL).\n"
            "Cocotb's official VPI bridge does not support Verilator on native Windows.\n\n"
            "To run Python co-simulations on Windows, please run inside WSL:\n"
            "  wsl python cli/main.py test-all-python\n\n"
            "Note: Native Windows fully supports Scala dynamic simulations and formal proofs:\n"
            "  python cli/main.py test-all\n"
            "  python cli/main.py test-all-formal\n",
            fg=typer.colors.YELLOW,
            bold=True
        )
        raise typer.Exit(code=1)

    from .python_runner import run_all_python_tests
    code = run_all_python_tests(
        filter_pattern=filter,
        fail_fast=fail_fast,
        log_dir=log_dir,
        dry_run=dry_run,
        verbose=verbose,
        ci_sleep=ci_sleep,
        debug_math=debug_math,
        extra_args=ctx.args if ctx.args else None
    )
    if code != 0:
        raise typer.Exit(code=code)

@app.command(name="test-python", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test_python_alias(
    ctx: typer.Context,
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter tests by name pattern (regex or substring)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show verbose output (-s, -v)"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory to store failure logs (default: out/python_reports)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List discovered Python test files without running them"),
    debug_math: bool = typer.Option(False, "--debug-math", help="Generate true_math_errors.log with detailed precision errors"),
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test file sequences (0 = disabled)"),
):
    """Alias for test-all-python."""
    test_all_python(
        ctx=ctx,
        filter=filter,
        fail_fast=fail_fast,
        verbose=verbose,
        log_dir=log_dir,
        dry_run=dry_run,
        debug_math=debug_math,
        ci_sleep=ci_sleep
    )

@app.command()
def build(
    src: Optional[Path] = typer.Argument(None, help="Source Verilog directory, single .v file, or .scala model [default: rtl/]"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output directory for build artifacts [default: hw_build/<board>/]"),
    board: str = typer.Option("tang-primer-20k", "--board", help="Target FPGA board profile from boards/*.json"),
    cst: Optional[Path] = typer.Option(None, "--cst", "--constraints", help="Custom physical constraints file override (.cst)"),
    top: Optional[str] = typer.Option(None, "--top", help="Top-level module name (auto-detected if omitted: UartSoC, top)"),
    synth_only: bool = typer.Option(False, "--synth-only", "--yosys", help="Stop after Yosys synthesis (quick resource check)"),
    pnr_only: bool = typer.Option(False, "--pnr-only", "--nextpnr", help="Stop after nextpnr place-and-route (skip bitstream pack)"),
    clk: Optional[str] = typer.Option(None, "--clk", help="Clock frequency override (e.g. '27MHz', '50MHz', '100MHz')"),
    no_dsp: bool = typer.Option(False, "--no-dsp", help="Disable hardware DSP block inference in synthesis (forces all arithmetic to LUTs)")
):
    """
    Synthesize, place & route and package FPGA bitstream (Yosys -> nextpnr -> gowin_pack).
    Accepts a Verilog directory, single .v file, or a .scala model file (auto-compiles to Verilog first).
    """
    from .build_runner import run_build
    code = run_build(
        src=src,
        out_dir=out,
        board=board,
        cst_override=cst,
        top_name=top,
        synth_only=synth_only,
        pnr_only=pnr_only,
        clk_override=clk,
        no_dsp=no_dsp
    )
    if code != 0:
        raise typer.Exit(code=code)

@app.command()
def flash(
    bitstream: Optional[Path] = typer.Argument(None, help="Path to bitstream (.fs, .bit). If omitted, auto-detected in hw_build/<board>/"),
    board: str = typer.Option("tang-primer-20k", "--board", help="Target FPGA board profile from boards/*.json"),
    sram: bool = typer.Option(True, "--sram/--no-sram", help="Flash to volatile SRAM (fast load, ~1s) [default: true]"),
    flash_mem: bool = typer.Option(False, "--flash", help="Flash to non-volatile on-board SPI flash memory")
):
    """
    Program the FPGA hardware using openFPGALoader.
    Auto-detects the latest bitstream in hw_build/<board>/ or takes an explicit bitstream path.
    """
    from .flash_runner import run_flash
    code = run_flash(
        bitstream=bitstream,
        board=board,
        sram=sram,
        flash_mem=flash_mem
    )
    if code != 0:
        raise typer.Exit(code=code)
