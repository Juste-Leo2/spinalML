import os
import sys
import glob
import random
import shutil
import subprocess
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SEED = 42

def seed_random(seed=None):
    """Seed the random module for reproducible tests (overridable via SPINALML_SEED)."""
    random.seed(seed if seed is not None else int(os.environ.get("SPINALML_SEED", SEED)))

def run_mill(test_class, dtype_filter, toplevel):
    os.makedirs(f"sim_build/{toplevel.lower()}", exist_ok=True)
    
    is_win = sys.platform == "win32"
    tools_dir = os.path.expanduser("~/.spinalml_tools")
    tools_mill = os.path.join(tools_dir, "mill.bat" if is_win else "mill")

    # Check local mill, spinalml_tools mill, or global PATH mill
    if os.path.exists("./mill"):
        mill_cmd = ["./mill.bat"] if is_win else ["bash", "./mill"]
    elif os.path.exists(tools_mill):
        mill_cmd = [tools_mill] if is_win else ["bash", tools_mill]
    else:
        global_mill = shutil.which("mill.bat" if is_win else "mill") or shutil.which("mill")
        if global_mill:
            mill_cmd = [global_mill] if is_win else ["bash", global_mill]
        else:
            mill_cmd = ["mill"]
            
    v_file = os.path.join(PROJECT_ROOT, f"{toplevel}.v")
    # Clean previous stale files if any
    if os.path.exists(v_file):
        try:
            os.remove(v_file)
        except OSError:
            pass

    cmd = mill_cmd + ["--no-server", "--disable-ticker", "spinalML.test.testOnly", f"{test_class}", "--", "-z", dtype_filter]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Mill compilation failed:\n{result.stdout}\n{result.stderr}")
        
    if not os.path.exists(v_file):
        # In Mill 0.11+ / 1.x, tests run in out/spinalML/test/testOnly.dest/sandbox/
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "out", "spinalML", "test", "**", f"{toplevel}.v"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(PROJECT_ROOT, "out", "**", f"{toplevel}.v"), recursive=True)
            
        if candidates:
            # Sort newest first
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            newest_v = candidates[0]
            shutil.copy(newest_v, v_file)
            
            # Also copy any associated ROM binary files (.bin) from the sandbox to PROJECT_ROOT
            src_dir = os.path.dirname(newest_v)
            for bin_file in glob.glob(os.path.join(src_dir, "*.bin")):
                shutil.copy(bin_file, os.path.join(PROJECT_ROOT, os.path.basename(bin_file)))

    if not os.path.exists(v_file):
        raise Exception(f"Verilog file {v_file} not found! Mill stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return v_file

@pytest.fixture(autouse=True)
def cleanup_verilog():
    yield
    # Nettoyage à la racine pour éviter la pollution
    for f in glob.glob(os.path.join(PROJECT_ROOT, "*.v")) + glob.glob(os.path.join(PROJECT_ROOT, "*.bin")):
        try:
            os.remove(f)
        except OSError:
            pass

def copy_roms(build_dir):
    """Copie les ROMs générées par SpinalHDL dans le dossier de simulation Verilator"""
    os.makedirs(build_dir, exist_ok=True)
    # Check PROJECT_ROOT first
    for f in glob.glob(os.path.join(PROJECT_ROOT, "*.bin")):
        shutil.copy(f, build_dir + "/")
    # Also check sandbox if any
    sandbox_bins = glob.glob(os.path.join(PROJECT_ROOT, "out", "spinalML", "test", "**", "*.bin"), recursive=True)
    for f in sandbox_bins:
        shutil.copy(f, os.path.join(build_dir, os.path.basename(f)))
