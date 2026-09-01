import subprocess
import sys
import typer
from typing import List

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

def run_tool(tool_name: str, args: List[str]):
    """Helper to run an installed tool and pass along arguments."""
    bin_path = get_bin_path(tool_name)
    
    if not bin_path.exists():
        typer.echo(f"Error: {tool_name} is not installed at {bin_path}.", err=True)
        typer.echo("Please run 'spinalml setup' first.", err=True)
        raise typer.Exit(code=1)
    
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
    
    # Run the command, replace the current process (cross-platform approach via subprocess)
    try:
        result = subprocess.run(cmd, cwd=str(CLI_DIR.parent))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        typer.echo(f"Error executing {tool_name}: {e}", err=True)
        sys.exit(1)

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
