import os
import re
import subprocess
import sys
import time
import typer
from typing import List, Optional
from pathlib import Path

from .config import load_config, get_bin_path, get_project_root, get_active_framework_root
from .installer import setup_tools
from .compile_flow import resolve_rounding as _resolve_rounding

app = typer.Typer(
    help="SpinalML CLI - Wrapper for FPGA Tools",
    context_settings={"help_option_names": ["-h", "--help"]}
)

@app.command()
def setup(
    debug: bool = typer.Option(False, "--debug", help="Show verbose raw logs"),
    force: bool = typer.Option(False, "-f", "--force", help="Force reinstallation of tools even if already up to date"),
    clean_cache: bool = typer.Option(False, "--clean-cache", help="Clean Coursier & Ivy caches before setup"),
    dev: bool = typer.Option(False, "--dev", help="Also install co-simulation extras (cocotb) into the managed Python env")
):
    """
    Download and extract all necessary tools (Mill, OSS CAD Suite, uv) to ~/.spinalml_tools,
    then (re)create the uv-managed Python envs (root .venv for the CLI, ~/.spinalml_tools/.venv
    for every Python flow). Plain setup leaves cocotb out; --dev adds it.
    """
    config = load_config()
    setup_tools(config, debug=debug, force=force, clean_cache=clean_cache, dev=dev)

@app.command()
def doctor():
    """
    Check the health of the uv-managed Python envs (interpreter, pins, leaks).
    """
    from .pyenv import print_doctor
    from rich.console import Console
    ok = print_doctor(console=Console())
    if not ok:
        raise typer.Exit(code=1)

@app.command(name="pylibs")
def pylibs():
    """
    Show pinned vs installed Python packages for each managed env.
    """
    from .pyenv import print_pylibs
    from rich.console import Console
    ok = print_pylibs(console=Console())
    if not ok:
        raise typer.Exit(code=1)

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
        work_dir = str(get_active_framework_root() if tool_name == "mill" else get_project_root())
        result = subprocess.run(cmd, cwd=work_dir)
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
def uv(ctx: typer.Context):
    """
    Run uv (managed Python package manager)
    """
    run_tool("uv", ctx.args)

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
    no_dsp: bool = typer.Option(False, "--no-dsp", help="Disable hardware DSP block inference (forces all arithmetic to LUTs)"),
    rounding: Optional[str] = typer.Option(None, "--rounding", help="Narrowing rounding policy for elaboration: 'rne' (default, unbiased) or 'trunc' (legacy bit-exact). If omitted, SPINALML_ROUNDING is kept, then RNE")
):
    """
    Compile a Scala file into Verilog by running it within the workspace module,
    and generate the supplementary hardware chain files (UartRx, UartTx, UartBridge,
    AxiReadMem) into the output directory with FPGA board-specific settings.
    """
    from .compile_flow import run_compile
    run_compile(file, out, chain, soc, board, clk, baud, out_count,
                word_width, bram_words, no_dsp, rounding, run_tool)


def _run_single_test_file(
    target_file: Path,
    stress: bool = False,
    stress_level: str = "heavy",
    stress_seed: int = 1,
) -> int:
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
        framework_root = get_active_framework_root()
        test_temp_dir = framework_root / "spinalML" / "test" / "src" / "cli_test_temp"
        if test_temp_dir.exists():
            shutil.rmtree(test_temp_dir)
        test_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # If the file is located outside the spinalML source tree (e.g. at repo root or tests/universal),
            # copy it into test_temp_dir so Mill automatically compiles it alongside the test scaffold.
            is_in_source = False
            try:
                target_file.resolve().relative_to((framework_root / "spinalML" / "src").resolve())
                is_in_source = True
            except ValueError:
                try:
                    target_file.resolve().relative_to((framework_root / "spinalML" / "test" / "src").resolve())
                    is_in_source = True
                except ValueError:
                    pass

            if not is_in_source:
                if not pkg:
                    # Model has no package header; inject 'package cli_test_temp' so Scala allows
                    # seamless access within the test harness without invalid _root_ imports.
                    raw_content = target_file.read_text(encoding="utf-8")
                    (test_temp_dir / target_file.name).write_text(f"package cli_test_temp\n\n{raw_content}", encoding="utf-8")
                else:
                    shutil.copy(target_file, test_temp_dir / target_file.name)

            import_stmt = f"import {pkg}.{comp_name}" if pkg else ""
            if stress and stress_level not in ("light", "heavy"):
                typer.echo(f"Error: --stress-level must be light or heavy (got {stress_level}).", err=True)
                raise typer.Exit(code=2)
            # T2: under --stress the sim top is a ChaosDut wrapper (DUT +
            # DramChaosInterposer + exposed ports); the oracle code below runs
            # unchanged against wrapper.dut, and runStress shares run's engine.
            wrapper_code = ""
            compile_open = "    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile {\n      val dut = new " + comp_name + "()\n"
            compile_close = "      dut\n    }"
            run_call = (
                "    val output = UniversalTestHarness.run(\n"
                "      compiled = compiled,\n"
                "      weightsWords = packedWords,\n"
                "      imageWords = imgWords,\n"
                "      expectedLogits = Some(replicaLogits),\n"
                "      imgBase = dutImgBase,\n"
                "      weightBase = dutWeightBase,\n"
                "      timeoutCycles = if (dutSpill) 200000 else 50000\n"
                "    )"
            )
            if stress:
                wrapper_code = (
                    "class StressWrapper extends Component with ChaosDut {\n"
                    f"  val dut = new {comp_name}()\n"
                    "  val axiCfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)\n"
                    f"  val chaos = DramChaosInterposer(axiCfg, DramChaosConfig.{stress_level}({stress_seed}L))\n"
                    "  val io = new Bundle {\n"
                    "    val memPort = master(Axi4(axiCfg))\n"
                    "    val ctrlBus = slave(AxiLite4(dut.axiLiteConfig))\n"
                    "    val outStream = master(cloneOf(dut.io.outStream))\n"
                    "  }\n"
                    "  chaos.io.dutSide <> dut.io.axiMaster\n"
                    "  chaos.io.memSide <> io.memPort\n"
                    "  io.ctrlBus <> dut.io.ctrlBus\n"
                    "  io.outStream <> dut.io.outStream\n"
                    "  def memAxi: Axi4 = io.memPort\n"
                    "  def ctrlAxi: AxiLite4 = io.ctrlBus\n"
                    "  def outShape: Seq[Int] = dut.io.outStream.shape\n"
                    "  def outLanes: Int = dut.io.outStream.lanes\n"
                    "  def setOutReady(v: Boolean): Unit = io.outStream.stream.ready #= v\n"
                    "  def outValid: Boolean = io.outStream.stream.valid.toBoolean\n"
                    "  def outPayload(lane: Int): Data = io.outStream.stream.payload(lane)\n"
                    "}\n\n"
                )
                compile_open = (
                    "    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile {\n"
                    "      val wrapper = new StressWrapper()\n"
                    "      val dut = wrapper.dut\n"
                )
                compile_close = "      wrapper\n    }"
                run_call = (
                    "    val output = UniversalTestHarness.runStress(\n"
                    "      compiled = compiled,\n"
                    "      weightsWords = packedWords,\n"
                    "      imageWords = imgWords,\n"
                    "      expectedLogits = Some(replicaLogits),\n"
                    "      imgBase = dutImgBase,\n"
                    "      weightBase = dutWeightBase,\n"
                    "      timeoutCycles = if (dutSpill) 200000 else 50000,\n"
                    f"      tag = \"{comp_name}-stress-{stress_level}-{stress_seed}\"\n"
                    "    )"
                )
            scaffold_code = f"""package cli_test_temp

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axi.{{Axi4Config, Axi4}}
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinalML.harness.{{MemoryHarness, UniversalTestHarness, ChaosDut, DramChaosConfig, DramChaosInterposer}}
import spinalML.replica.{{HWArithmetic, WeightMemoryLayout, ModelReplica}}
import spinalML.nn.{{Accelerator, Linear, MemorySpec}}
{import_stmt}

{wrapper_code}class AutoGeneratedCircuitTest extends AnyFunSuite {{
  test("{comp_name} universal bit-exact circuit verification under Verilator") {{
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    
    var replicaLogits: Seq[Double] = null
    var packedWords: Seq[BigInt] = null
    var imgWords: Seq[BigInt] = null
    // R4: DUT memory map (defaults = historical hard-coded bases, so every
    // existing model behaves exactly as before). The future `--stress` flag
    // will plug in here as an extra harness arg (timing only, oracle untouched).
    var dutImgBase: Long = 0x10000L
    var dutWeightBase: Long = 0x20000L
    var dutSpill: Boolean = false

{compile_open}      val layers = dut.modelSpec
      val inputShape = dut.inputShape
      val pipelineDtype = dut.globalDataType
      val memSpec: MemorySpec = dut match {{
        case a: Accelerator[_] => a.memory
        case _ => MemorySpec.default
      }}
      dutImgBase = memSpec.imgBase
      dutWeightBase = memSpec.weightBase
      dutSpill = layers.exists {{ case l: Linear if l.spilling => true; case _ => false }}
      
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
        val (eW: Int, mW: Int) = inData match {{
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
{compile_close}
    
{run_call}
    
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
    ci_sleep: float = typer.Option(0.0, "--ci", help="Pause in seconds between test sequences (0 = disabled; use e.g. --ci 3 on slow-SD/self-hosted runners)"),
    stress: bool = typer.Option(False, "--stress", help="Run behind the DRAM chaos interposer (timing pressure only, bit-exact oracle unchanged)"),
    stress_level: str = typer.Option("heavy", "--stress-level", help="Chaos intensity when --stress is set: light or heavy"),
    stress_seed: int = typer.Option(1, "--stress-seed", help="Deterministic chaos seed when --stress is set"),
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
            rc = _run_single_test_file(f, stress=stress, stress_level=stress_level, stress_seed=stress_seed)
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

    rc = _run_single_test_file(file, stress=stress, stress_level=stress_level, stress_seed=stress_seed)
    if rc != 0:
        raise typer.Exit(code=rc)

@app.command(name="test-all", hidden=True)
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

@app.command(name="test-all-formal", hidden=True)
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

@app.command(name="test-all-python", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
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
    no_dsp: bool = typer.Option(False, "--no-dsp", help="Disable hardware DSP block inference in synthesis (forces all arithmetic to LUTs)"),
    rounding: Optional[str] = typer.Option(None, "--rounding", help="Narrowing rounding policy for elaboration: 'rne' (default, unbiased) or 'trunc' (legacy bit-exact). If omitted, SPINALML_ROUNDING is kept, then RNE")
):
    """
    Synthesize, place & route and package FPGA bitstream (Yosys -> nextpnr -> gowin_pack).
    Accepts a Verilog directory, single .v file, or a .scala model file (auto-compiles to Verilog first).
    """
    from .build_runner import run_build
    import os as _os
    rounding_value, rounding_label = _resolve_rounding(rounding)
    if rounding_value is not None:
        _os.environ["SPINALML_ROUNDING"] = rounding_value
    typer.echo(f"Rounding       : {rounding_label}")
    code = run_build(
        src=src,
        out_dir=out,
        board=board,
        cst_override=cst,
        top_name=top,
        synth_only=synth_only,
        pnr_only=pnr_only,
        clk_override=clk,
        no_dsp=no_dsp,
        rounding=rounding
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
