import subprocess
import sys
import typer
from typing import List, Optional
from pathlib import Path

from .config import load_config, get_bin_path, CLI_DIR
from .installer import setup_tools

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
    out: Path = typer.Option(None, "-o", "--out", help="Output directory for generated Verilog files")
):
    """
    Compile a Scala file into Verilog by running it within the workspace module.
    """
    import shutil
    import os
    import glob
    
    if not file.exists():
        typer.echo(f"Error: File {file} does not exist.", err=True)
        raise typer.Exit(code=1)
        
    import re
    import shutil
    import glob
    
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
    
    if app_match:
        main_class = app_match.group(1)
        full_main = f"{pkg}.{main_class}" if pkg else main_class
    else:
        # Try to find a Component or Accelerator
        comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
        if not comp_match:
            typer.echo(f"Error: {file.name} does not contain 'object <Name> extends App' nor a Component.", err=True)
            typer.echo("Please add an App entry point to generate Verilog.", err=True)
            shutil.rmtree(workspace_src)
            raise typer.Exit(code=1)
            
        comp_name = comp_match.group(1)
        import_stmt = f"import {pkg}.{comp_name}" if pkg else ""
        target_dir = str(out.resolve()).replace('\\', '/') if out else "."
        auto_runner_code = f"""
package spinalml_auto
import spinal.core._
{import_stmt}

object AutoRunner extends App {{
  SpinalConfig(
    targetDirectory = "{target_dir}",
    headerWithDate = true,
    rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
  ).generateVerilog(new {comp_name}())
}}
"""
        (workspace_src / "AutoRunner.scala").write_text(auto_runner_code, encoding="utf-8")
        full_main = "spinalml_auto.AutoRunner"
        auto_generated = True
        typer.echo(f"Auto-generating runner for component {comp_name}...")
    
    project_root = CLI_DIR.parent
    existing_v_files = set(glob.glob(str(project_root / "*.v")))
    
    typer.echo(f"Running Mill spinalML.runMain {full_main}...")
    ret_code = run_tool("mill", ["spinalML.runMain", full_main], exit_on_error=False)
        
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
        
    # 4. Move generated .v files if --out is specified (for non-auto-generated or fallback)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        # Check if the component verilog exists in project_root and move it
        comp_v = project_root / f"{comp_name}.v" if not app_match else None
        if comp_v and comp_v.exists():
            shutil.move(str(comp_v), str(out / comp_v.name))
            typer.echo(f"Moved generated {comp_v.name} to {out}")
        else:
            current_v_files = set(glob.glob(str(project_root / "*.v")))
            new_v_files = current_v_files - existing_v_files
            for v_file in new_v_files:
                dest_v = out / Path(v_file).name
                shutil.move(v_file, dest_v)
                typer.echo(f"Moved generated {Path(v_file).name} to {out}")
            if not new_v_files and not (out / f"{comp_name}.v" if not app_match else False).exists():
                typer.echo(f"Generated Verilog files are in {out}")
    else:
        typer.echo("Compilation complete. (Verilog files are in the project root)")


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
        ret_code = run_tool("mill", ["spinalML.test.testOnly", full_test], exit_on_error=False)
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

            ret_code = run_tool("mill", ["spinalML.test.testOnly", full_test], exit_on_error=False)
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
        ret_code = run_tool("mill", ["spinalML.test.runMain", full_test], exit_on_error=False)
        if ret_code == 0:
            typer.echo(f"Test {test_obj} passed successfully!")
        return ret_code

    typer.echo(f"Error: Could not identify a test suite or Component/Accelerator in {target_file.name}.", err=True)
    return 1


@app.command()
def test(
    file: Path = typer.Argument(..., help="Path to the Scala model or test file/directory to run")
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
):
    """
    Run all ScalaTest suites sequentially (1-by-1) to avoid Verilator/G++ RAM exhaustion.
    Captures failure traces into individual log files under out/test_reports/.
    """
    from .test_runner import run_all_tests
    code = run_all_tests(filter_pattern=filter, fail_fast=fail_fast, log_dir=log_dir, dry_run=dry_run, verbose=verbose)
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
):
    """
    Run all SymbiYosys formal verification suites sequentially (1-by-1).
    Discovers all *Formal.scala specs under symbolicTest/ and executes them via SMT-BMC.
    """
    from .formal_runner import run_all_formal_tests
    code = run_all_formal_tests(filter_pattern=filter, fail_fast=fail_fast, log_dir=log_dir, dry_run=dry_run, timeout=timeout, verbose=verbose)
    if code != 0:
        raise typer.Exit(code=code)

@app.command(name="test-all-python", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test_all_python(
    ctx: typer.Context,
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter tests by name pattern (pytest -k)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show verbose output (-s, -v)"),
    debug_math: bool = typer.Option(False, "--debug-math", help="Generate true_math_errors.log with detailed precision errors"),
):
    """
    Run Python Cocotb/Pytest hardware co-simulations with Verilator.
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

    from .test_runner import setup_tool_env
    env = setup_tool_env()
    
    cmd = [sys.executable, "-m", "pytest", "tests/python"]
    if filter:
        cmd.extend(["-k", filter])
    if fail_fast:
        cmd.append("-x")
    if verbose:
        cmd.extend(["-s", "-v"])
    else:
        cmd.append("-s")
    if debug_math:
        cmd.append("--debug-math")
        
    if ctx.args:
        cmd.extend(ctx.args)
        
    typer.echo(f"Running Python hardware co-simulations: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(CLI_DIR.parent), env=env)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)

@app.command(name="test-python", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test_python_alias(
    ctx: typer.Context,
    filter: Optional[str] = typer.Option(None, "-k", "--filter", help="Filter tests by name pattern (pytest -k)"),
    fail_fast: bool = typer.Option(False, "-x", "--fail-fast", help="Stop execution immediately on first failure"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show verbose output (-s, -v)"),
    debug_math: bool = typer.Option(False, "--debug-math", help="Generate true_math_errors.log with detailed precision errors"),
):
    """Alias for test-all-python."""
    test_all_python(ctx=ctx, filter=filter, fail_fast=fail_fast, verbose=verbose, debug_math=debug_math)
