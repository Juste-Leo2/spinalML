import os
import glob
import shutil
import subprocess
import pytest

def run_mill(test_class, dtype_filter, toplevel):
    os.makedirs(f"sim_build/{toplevel.lower()}", exist_ok=True)
    
    # Check if local mill exists, otherwise fallback to global mill for Radxa compatibility
    if os.path.exists("./mill"):
        mill_cmd = ["bash", "./mill"]
    else:
        mill_cmd = ["mill"]
        
    cmd = mill_cmd + ["spinalML.test.testOnly", f"{test_class}", "--", "-z", dtype_filter]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/home/leonard/spinalML", capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Mill compilation failed:\n{result.stdout}\n{result.stderr}")
        
    v_file = f"/home/leonard/spinalML/{toplevel}.v"
    if not os.path.exists(v_file):
        raise Exception(f"Verilog file {v_file} not found!")
    return v_file

@pytest.fixture(autouse=True)
def cleanup_verilog():
    yield
    # Nettoyage à la racine pour éviter la pollution
    for f in glob.glob("/home/leonard/spinalML/*.v") + glob.glob("/home/leonard/spinalML/*.bin"):
        try:
            os.remove(f)
        except OSError:
            pass

def copy_roms(build_dir):
    """Copie les ROMs générées par SpinalHDL dans le dossier de simulation Verilator"""
    os.makedirs(build_dir, exist_ok=True)
    for f in glob.glob("/home/leonard/spinalML/*.bin"):
        shutil.copy(f, build_dir + "/")
