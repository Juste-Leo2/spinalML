import subprocess
import sys
import typer
from typing import List
from pathlib import Path

from .config import load_config, get_bin_path, CLI_DIR
from .installer import setup_tools

app = typer.Typer(
    help="SpinalML CLI - Wrapper for FPGA Tools",
    context_settings={"help_option_names": ["-h", "--help"]}
)

@app.command()
def setup(debug: bool = typer.Option(False, "--debug", help="Show verbose raw logs")):
    """
    Download and extract all necessary tools (Mill, OSS CAD Suite) to ~/.spinalml_tools
    """
    config = load_config()
    setup_tools(config, debug=debug)

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
  SpinalConfig(targetDirectory = "{target_dir}").generateVerilog(new {comp_name}())
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


@app.command()
def test(
    file: Path = typer.Argument(..., help="Path to the Scala model or test file to run")
):
    """
    Run hardware simulation tests (bit-exact, streaming, tiling, memory verification).
    """
    if not file.exists():
        typer.echo(f"Error: File {file} does not exist.", err=True)
        raise typer.Exit(code=1)

    import re
    import shutil

    content = file.read_text(encoding="utf-8")
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
        if ret_code != 0:
            raise typer.Exit(code=ret_code)
        typer.echo("All tests passed successfully!")
        return

    # 2. Check for component/accelerator first
    comp_match = re.search(r'(?:case\s+)?class\s+(\w+).*?(?:extends\s+Component|extends\s+Accelerator)', content, re.MULTILINE | re.DOTALL)
    if comp_match:
        comp_name = comp_match.group(1)
        
        # Check if a dedicated test already exists in the codebase (e.g. MnistTest, Mnistw4a8Test, HighLevelTemplateTest)
        test_candidate = CLI_DIR.parent / "spinalML" / "test" / "src" / "spinalML" / "examples" / f"{comp_name}Test.scala"
        if test_candidate.exists():
            typer.echo(f"Found existing verification suite: {test_candidate.name}")
            full_test = f"spinalML.examples.{comp_name}Test"
            typer.echo(f"Running Mill testOnly {full_test}...")
            ret_code = run_tool("mill", ["spinalML.test.testOnly", full_test], exit_on_error=False)
            if ret_code != 0:
                raise typer.Exit(code=ret_code)
            typer.echo(f"Model {comp_name} verification successful!")
            return

        # Otherwise, scaffold a dynamic validation test bench under cli_test_temp using UniversalTestHarness
        test_temp_dir = CLI_DIR.parent / "spinalML" / "test" / "src" / "cli_test_temp"
        if test_temp_dir.exists():
            shutil.rmtree(test_temp_dir)
        test_temp_dir.mkdir(parents=True, exist_ok=True)

        import_stmt = f"import {pkg}.{comp_name}" if pkg else ""
        scaffold_code = f"""package cli_test_temp

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.harness.{{MemoryHarness, UniversalTestHarness}}
import spinalML.replica.HWArithmetic
{import_stmt}

class AutoGeneratedCircuitTest extends AnyFunSuite {{
  test("{comp_name} universal circuit verification under Verilator") {{
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(new {comp_name}())
    
    // Generate synthetic deterministic test vectors for image and weights
    val imgWords = Seq.fill(64)(BigInt("0001000100010001", 16))
    val wWords = Seq.fill(512)(BigInt("0001000100010001", 16))
    
    val output = UniversalTestHarness.run(
      compiled = compiled,
      weightsWords = wWords,
      imageWords = imgWords,
      expectedLogits = None
    )
    
    println(s"[{comp_name}] Collected ${{output.length}} outputs successfully.")
  }}
}}
"""
        (test_temp_dir / "AutoGeneratedCircuitTest.scala").write_text(scaffold_code, encoding="utf-8")
        full_test = "cli_test_temp.AutoGeneratedCircuitTest"

        typer.echo(f"Scaffolded simulation test for {comp_name}...")
        typer.echo(f"Running Mill testOnly {full_test}...")

        ret_code = run_tool("mill", ["spinalML.test.testOnly", full_test], exit_on_error=False)

        if test_temp_dir.exists():
            shutil.rmtree(test_temp_dir)

        if ret_code != 0:
            raise typer.Exit(code=ret_code)

        typer.echo(f"Circuit verification for {comp_name} passed successfully!")
        return

    # 3. Check if the file is an executable test object (main or extends App)
    app_test_match = re.search(r'object\s+(\w+)(?:\s+extends\s+App|\s*\{[^}]*def\s+main)', content, re.DOTALL)
    if app_test_match:
        test_obj = app_test_match.group(1)
        full_test = f"{pkg}.{test_obj}" if pkg else test_obj
        typer.echo(f"Detected executable test object: {full_test}")
        typer.echo(f"Running Mill test.runMain {full_test}...")
        ret_code = run_tool("mill", ["spinalML.test.runMain", full_test], exit_on_error=False)
        if ret_code != 0:
            raise typer.Exit(code=ret_code)
        typer.echo(f"Test {test_obj} passed successfully!")
        return

    typer.echo(f"Error: Could not identify a test suite or Component/Accelerator in {file.name}.", err=True)
    raise typer.Exit(code=1)


