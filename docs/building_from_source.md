# Setting up SpinalML from Source & Running All Test Suites

This document is the complete reference for setting up the full **SpinalML** development environment from source, running the entire hardware verification matrix (ScalaTest, SymbiYosys Formal, and Python Cocotb co-simulations), and building standalone executables.

> [!IMPORTANT]
> **Working Directory Rule:**
> All commands in this guide must **always be executed from the repository root** (`spinalML/`), never from subdirectories.

---

## 1. Prerequisites & System Dependencies

### Platform Support:
- **Scala simulations, compilation, synthesis, and FPGA flashing**: Fully supported natively on **Linux, macOS, and Windows**.
- **Python Cocotb co-simulations (`test-all-python`)**: Requires **Linux** (native or via **WSL 2 on Windows**). Cocotb's VPI bridge with Verilator is not supported on native Windows.

### System Packages (Linux / Ubuntu / WSL 2):
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    default-jdk curl build-essential git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
```

---

## 2. The Python 3.12 Global Requirement (Crucial for Cocotb)

> [!WARNING]
> **VPI Teardown Safety Gate**:
> Cocotb hardware co-simulation with Verilator requires a **Python 3.12** environment. If the global interpreter or the `.venv` interpreter does not match Python 3.12 (e.g. running on an older system Python like 3.10 on Ubuntu 22.04), the simulation will encounter memory corruption upon exit (`double free or corruption` during VPI bridge teardown).

### Step 2.1: Ensure Global Python is 3.12 via `pyenv`
If your distribution does not have Python 3.12 as default, use `pyenv`:

```bash
# 1. Install pyenv if missing
if ! command -v pyenv &> /dev/null; then
  curl -LsSf https://pyenv.run | bash
fi

# 2. Add pyenv to your shell environment
export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
eval "$(pyenv init -)"

# 3. Install and set Python 3.12 as global
pyenv install -s 3.12.1
pyenv global 3.12.1
pyenv rehash

# 4. Verify that global python3 reports 3.12.x
python3 --version
```

---

## 3. Environment Bootstrap with `uv`

We use [**uv**](https://github.com/astral-sh/uv) for fast, deterministic dependency management.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Clone the repository and enter the root
git clone https://github.com/Juste-Leo2/spinalML.git
cd spinalML

# 3. Create .venv explicitly locked to Python 3.12
uv venv --python 3.12.1 --clear .venv
source .venv/bin/activate

# 4. Install all development dependencies (CLI + pytest + cocotb + numpy)
uv pip install -r requirements.txt
```

---

## 4. Hardware Toolchain Provisioning (`setup`)

SpinalML automates the installation of Mill, Verilator, SymbiYosys, Yosys, and nextpnr inside `~/.spinalml_tools`, leaving your global system clean:

```bash
# Run from repository root
python cli/main.py setup
```

To verify the installation:
```bash
python cli/main.py verilator --version
python cli/main.py yosys -V
python cli/main.py mill version
```

---

## 5. Running the Verification Matrix

With the full development environment active, you can execute all test suites:

### 1. ScalaTest Dynamic Simulations (75 Suites)
Runs sequentially to avoid Verilator/G++ RAM exhaustion:
```bash
python cli/main.py test-all -v
```

### 2. Universal Bit-Exact Verification (Engine Check)
Verifies cycle-accurate bit-exact match against software arithmetic:
```bash
python cli/main.py test tests/universal/UniversalOpsDemo.scala
```

### 3. SymbiYosys Formal Proofs (56 Suites)
Executes bounded model checking (BMC) and induction proofs:
```bash
python cli/main.py test-all-formal -v
```

### 4. Python Cocotb Co-Simulations
Executes test cases in `tests/python/` via pytest & Cocotb:
```bash
python cli/main.py test-all-python -v
```

---

## 6. Building the Standalone CLI Executable

If you want to package your modified SpinalML CLI into a single-file executable:

```bash
# 1. Install PyInstaller
uv pip install pyinstaller

# 2. Run the build script
python cli/build_binary.py
```

The resulting standalone binary will be created in `dist/` (e.g. `dist/spinalml-linux-x64` or `dist/spinalml-windows-x64.exe`). It embeds the entire `spinalML/` library, `build.mill`, and `boards/` in a compact ~15 MB executable.
