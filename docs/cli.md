# SpinalML Command Line Interface (CLI)

The **SpinalML CLI** (`cli/main.py`) provides an integrated, cross-platform toolchain management and verification suite for developing, compiling, and testing machine learning hardware accelerators in SpinalHDL.

It manages external EDA dependencies (Mill, Verilator, SymbiYosys, Yosys, nextpnr, openFPGALoader) with zero manual environment pollution, automatically routing PATH and library variables.

---

## Table of Contents
1. [Environment Setup](#1-environment-setup)
   - [Linux](#linux)
   - [Windows](#windows)
   - [Toolchain Provisioning (`setup`)](#toolchain-provisioning-setup)
2. [Hardware RTL Compilation (`compile`)](#2-hardware-rtl-compilation-compile)
   - [Auto-Generating Verilog from Components](#auto-generating-verilog-from-components)
   - [Compiling Custom App Generators](#compiling-custom-app-generators)
3. [Circuit Simulation & Verification (`test`)](#3-circuit-simulation--verification-test)
   - [Universal Bit-Exact Hardware Verification (e.g. `Mnistw4a8`)](#universal-bit-exact-hardware-verification-eg-mnistw4a8)
   - [Executing Dedicated ScalaTest Suites](#executing-dedicated-scalatest-suites)
   - [Running Executable Test Objects](#running-executable-test-objects)
4. [Full Regression Testing (`test-all`)](#4-full-regression-testing-test-all)
5. [Formal Verification Engine (`test-all-formal`)](#5-formal-verification-engine-test-all-formal)
6. [Low-Level EDA Tool Passthroughs](#6-low-level-eda-tool-passthroughs)
7. [Quick Command Reference](#7-quick-command-reference)

---

## 1. Environment Setup

We recommend using [**uv**](https://docs.astral.sh/uv/), an extremely fast Python package and environment manager.

### Installing `uv`

#### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### Project Setup with `uv`

#### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/Juste-Leo2/spinalML.git
cd spinalML

# Create virtual environment with Python 3.11
uv venv -p 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

To run commands:
```bash
python cli/main.py --help
# Or seamlessly via uv without manual activation:
uv run python cli/main.py --help
```

---

#### Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/Juste-Leo2/spinalML.git
cd spinalML

# Create virtual environment with Python 3.11
uv venv -p 3.11
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt
```

To run commands:
```powershell
python cli/main.py --help
# Or seamlessly via uv without manual activation:
uv run python cli/main.py --help
```

---

### Toolchain Provisioning (`setup`)

The `setup` command automatically fetches and unpacks **Mill** (Scala build tool) and **OSS CAD Suite** (Verilator, SymbiYosys, Yosys, CVC4, Z3, nextpnr, openFPGALoader) into your user home folder (`~/.spinalml_tools/`):

```bash
# Linux
python cli/main.py setup

# Windows
python cli/main.py setup
```

Options:
- `--debug` : Displays verbose extraction and download logs.

---

## 2. Hardware RTL Compilation (`compile`)

The `compile` command elaborates SpinalHDL code and generates synthesizable Verilog (`.v`).

```bash
python cli/main.py compile <path_to_scala_file> [OPTIONS]
```

### Auto-Generating Verilog from Components

If your file defines a `Component` or `Accelerator` class without an entry point, the CLI automatically synthesizes an ephemeral runner, elaborates the design, and emits the Verilog:

```bash
# Linux
python cli/main.py compile spinalML/src/spinalML/examples/Mnist.scala -o verilog/

# Windows
python cli/main.py compile spinalML\src\spinalML\examples\Mnistw4a8.scala -o verilog\
```

### Compiling Custom App Generators

If your Scala file already contains an `object <Name> extends App` entry point, the CLI executes it directly:

```bash
python cli/main.py compile spinalML/src/spinalML/examples/SimpleCNN.scala
```

Options:
- `-o`, `--out <PATH>` : Specifies the destination directory for the generated Verilog file(s).

---

## 3. Circuit Simulation & Verification (`test`)

The `test` command runs hardware simulations using Verilator with cycle-accurate evaluation.

```bash
python cli/main.py test <path_to_scala_or_test_file>
```

### Universal Bit-Exact Hardware Verification (e.g. `Mnistw4a8`)

When pointed directly at an **accelerator model source file**, the CLI automatically activates the **Universal Verification Engine** (`UniversalTestHarness`):
1. Analyzes the model specification (`modelSpec`, `inputShape`).
2. Computes the golden software reference outputs using the bit-accurate model replica (`ModelReplica`).
3. Synthesizes memory layouts and compiles the circuit under **Verilator**.
4. Feeds packed stimulus streams over AXI4 and asserts bit-exact match against golden tensors.

```bash
# Linux
python cli/main.py test spinalML/src/spinalML/examples/Mnistw4a8.scala
python cli/main.py test spinalML/src/spinalML/examples/Mnist.scala

# Windows
python cli/main.py test spinalML\src\spinalML\examples\Mnistw4a8.scala
```

### Executing Dedicated ScalaTest Suites

When pointed at any ScalaTest suite (`extends AnyFunSuite`), the CLI invokes Mill `testOnly` on that exact class:

```bash
# Linux
python cli/main.py test spinalML/test/src/spinalML/examples/MnistTest.scala
python cli/main.py test spinalML/test/src/spinalML/ops/Conv2DTest.scala

# Windows
python cli/main.py test spinalML\test\src\spinalML\examples\Mnistw4a8Test.scala
```

### Running Executable Test Objects

When pointed at an executable test object (`extends App` or with a `def main`), the CLI runs it via Mill's `test.runMain`:

```bash
python cli/main.py test spinalML/test/src/spinalML/examples/SimplePipelineTest.scala
```

---

## 4. Full Regression Testing (`test-all`)

To prevent system memory exhaustion caused by parallel Verilator C++ compilations, `test-all` executes **all 75 discovered dynamic ScalaTest suites sequentially (1-by-1)**.

```bash
# Run the entire test suite sequentially
python cli/main.py test-all
```

Output summary:
```text
            Test Execution Summary
+-------------------------+--------------------+
| Metric                  | Value              |
+-------------------------+--------------------+
| Total Suites Discovered | 75                 |
| Suites Executed         | 75                 |
| Passed                  | 75                 |
| Failed                  | 0                  |
| Total Time              | 3615.7s (60.3 min) |
+-------------------------+--------------------+
All 75 tests passed successfully!
```

### Options & Filtering

* **Verbose failure output** (`-v`, `--verbose`):
  Prints full stdout and stderr directly in the terminal upon failure (essential for CI pipelines).
  ```bash
  python cli/main.py test-all -v
  ```
* **Filter by name pattern** (`-k`, `--filter`):
  ```bash
  # Run only quantization and pooling tests
  python cli/main.py test-all -k "Quant|Pool"
  ```
* **Stop on first failure** (`-x`, `--fail-fast`):
  ```bash
  python cli/main.py test-all --fail-fast
  ```
* **Dry run** (`--dry-run`): List all discovered test suites without executing them:
  ```bash
  python cli/main.py test-all --dry-run
  ```
* **Custom report directory** (`--log-dir`):
  ```bash
  python cli/main.py test-all --log-dir out/my_reports
  ```

---

## 5. Formal Verification Engine (`test-all-formal`)

SpinalML features an exhaustive formal verification suite using **SymbiYosys (SBY)** and **SMT-BMC (CVC4 / Z3)**. All 56 formal specifications (`*Formal.scala` under `symbolicTest/`) verify structural flow invariants, AXI4/AXI4-Stream handshakes, absence of deadlocks, and CSR registers.

```bash
# Run all 56 formal verification suites sequentially
python cli/main.py test-all-formal
```

Output summary:
```text
            Formal Verification Summary
+--------------------------------+--------------------+
| Metric                         | Value              |
+--------------------------------+--------------------+
| Total Formal Suites Discovered | 56                 |
| Suites Executed                | 56                 |
| Passed                         | 56                 |
| Failed                         | 0                  |
| Total Time                     | ~25 min            |
+--------------------------------+--------------------+
All 56 formal verification suites passed successfully!
```

### Options & Filtering

* **Verbose failure output** (`-v`, `--verbose`):
  Prints full solver traces directly in the terminal upon proof failure.
  ```bash
  python cli/main.py test-all-formal -v
  ```
* **Filter formal tests** (`-k`, `--filter`):
  ```bash
  # Run formal verification only on accelerator and memory components
  python cli/main.py test-all-formal -k "Accelerator|DMA|DoubleBuffer"
  ```
* **Adjust solver timeout** (`-t`, `--timeout`, default: 900s):
  ```bash
  python cli/main.py test-all-formal -t 600
  ```
* **Fail-fast mode** (`-x`, `--fail-fast`):
  ```bash
  python cli/main.py test-all-formal --fail-fast
  ```
* **Dry run** (`--dry-run`):
  ```bash
  python cli/main.py test-all-formal --dry-run
  ```

---

## 6. Low-Level EDA Tool Passthroughs

The CLI provides transparent wrappers around all bundled FPGA tools, automatically configuring `PATH`, `VERILATOR_ROOT`, and GCC toolchain paths:

### Mill (Scala Build Tool)
```bash
# Recompile project
python cli/main.py mill spinalML.compile

# Run a single formal test directly
python cli/main.py mill spinalML.test.runMain spinalML.symbolicTest.dtypes.FP4Formal
```

### Verilator
```bash
python cli/main.py verilator --version
```

### SymbiYosys (SBY)
```bash
python cli/main.py sby --help
```

### Yosys
```bash
python cli/main.py yosys -V
```

### nextpnr (Place & Route)
```bash
# Specify target architecture as first argument:
python cli/main.py nextpnr ice40 --help
python cli/main.py nextpnr ecp5 --help
```

### openFPGALoader (FPGA Flashing)
```bash
python cli/main.py openfpgaloader --detect
```

---

## 7. Quick Command Reference

| Action | Linux Command | Windows PowerShell Command |
| :--- | :--- | :--- |
| **Install `uv`** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Create venv** | `uv venv -p 3.11` | `uv venv -p 3.11` |
| **Install requirements** | `uv pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| **Activate venv** | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| **Install tools** | `python cli/main.py setup` | `python cli/main.py setup` |
| **Compile to Verilog** | `python cli/main.py compile spinalML/src/spinalML/examples/Mnist.scala -o verilog/` | `python cli/main.py compile spinalML\src\spinalML\examples\Mnist.scala -o verilog\` |
| **Universal Circuit Test** | `python cli/main.py test spinalML/src/spinalML/examples/Mnistw4a8.scala` | `python cli/main.py test spinalML\src\spinalML\examples\Mnistw4a8.scala` |
| **Single ScalaTest** | `python cli/main.py test spinalML/test/src/spinalML/examples/MnistTest.scala` | `python cli/main.py test spinalML\test\src\spinalML\examples\MnistTest.scala` |
| **Run All Dynamic Tests** | `python cli/main.py test-all` | `python cli/main.py test-all` |
| **Run Filtered Tests** | `python cli/main.py test-all -k "Conv2D"` | `python cli/main.py test-all -k "Conv2D"` |
| **Run All Formal Proofs** | `python cli/main.py test-all-formal` | `python cli/main.py test-all-formal` |
| **Run Filtered Formal** | `python cli/main.py test-all-formal -k "Accelerator"` | `python cli/main.py test-all-formal -k "Accelerator"` |
| **Direct Mill Command** | `python cli/main.py mill spinalML.compile` | `python cli/main.py mill spinalML.compile` |
