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
3. [Turnkey FPGA Synthesis & Packing (`build`)](#3-turnkey-fpga-synthesis--packing-build)
   - [Full Flow: Scala to Silicon Bitstream](#full-flow-scala-to-silicon-bitstream)
   - [Intermediate Stop Stages (`--yosys`, `--nextpnr`)](#intermediate-stop-stages---yosys---nextpnr)
   - [Board Profiles & Physical Pin Constraints](#board-profiles--physical-pin-constraints)
4. [FPGA Hardware Flashing & Deployment (`flash`)](#4-fpga-hardware-flashing--deployment-flash)
   - [SRAM Volatile Programming](#sram-volatile-programming)
   - [SPI Flash Non-Volatile Programming](#spi-flash-non-volatile-programming)
5. [Circuit Simulation & Verification (`test`)](#5-circuit-simulation--verification-test)
   - [Universal Bit-Exact Hardware Verification (e.g. `Mnistw4a8`)](#universal-bit-exact-hardware-verification-eg-mnistw4a8)
   - [Executing Dedicated ScalaTest Suites](#executing-dedicated-scalatest-suites)
   - [Running Executable Test Objects](#running-executable-test-objects)
6. [Full Regression Testing (`test-all`)](#6-full-regression-testing-test-all)
7. [Formal Verification Engine (`test-all-formal`)](#7-formal-verification-engine-test-all-formal)
8. [Python Hardware Co-Simulations (`test-all-python`)](#8-python-hardware-co-simulations-test-all-python)
9. [Low-Level EDA Tool Passthroughs](#9-low-level-eda-tool-passthroughs)
10. [Quick Command Reference](#10-quick-command-reference)

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
- `-o`, `--out <PATH>` : Destination directory for generated Verilog files [default: `rtl`].
- `--soc / --no-soc` : Generate complete turnkey `UartSoC` top-level wrapping the accelerator [default: `--no-soc`].
- `--chain / --no-chain` : Generate supplementary UART chain Verilog files (`UartRx`, `UartTx`, `UartBridge`, `AxiReadMem`) [default: `--chain`].
- `--board <NAME>` : Target FPGA board profile from `boards/*.json` [default: `tang-primer-20k`].
- `--clk <FREQ>` : Hardware clock frequency override (e.g. `'27MHz'`, `'50MHz'`, `'100MHz'`).
- `--baud <INT>` : UART baud rate override (e.g. `115200`, `921600`).
- `--out-count <INT>` : Number of output stream bytes/logits (auto-detected from model if omitted).
- `--word-width <INT>` : AXI data bus width in bits (auto-detected from model if omitted).
- `--bram-words <INT>` : BRAM capacity in 64-bit words (default: from board or 4096).

---

## 3. Turnkey FPGA Synthesis & Packing (`build`)

The `build` command provides a completely automated, end-to-end silicon compilation pipeline:
```
Scala Model (.scala) or Verilog (rtl/)
    │
    ▼ [Auto-Elaboration into UartSoC if .scala]
Turnkey RTL Netlist
    │
    ▼ [Phase 1: Yosys Synthesis]
Target Primitive Netlist (LUTs, FFs, BRAM, DSP)
    │
    ▼ [Phase 2: nextpnr-himbaechel PnR]
Physical Placement & Routing + Static Timing Analysis (Fmax)
    │
    ▼ [Phase 3: Bitstream Packing (gowin_pack / apycula)]
FPGA Bitstream (hw_build/<board>/top.fs)
```

### Full Flow: Scala to Silicon Bitstream

You can pass a `.scala` model directly to `build`. SpinalML will compile the model into Verilog, bundle the turnkey `UartSoC` (AXI4 Master, CSRs, BRAM, UART Bridge), run Yosys synthesis, place & route with nextpnr, and produce the bitstream in `hw_build/<board>/top.fs`:

```bash
# Full build from a Scala neural network specification
python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k

# Or build from an existing Verilog directory (defaults to rtl/)
python cli/main.py build rtl/ --board tang-primer-20k
```

### Intermediate Stop Stages (`--yosys`, `--nextpnr`)

For quick iteration and sanity checks without waiting for the full pipeline:

1. **Quick Synthesis & Logic Gate Estimation (`--yosys` / `--synth-only`)**:
   Stops immediately after Yosys, printing the gate count estimation without running place & route.
   ```bash
   python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k --yosys
   ```

2. **Placement, Routing & Timing Analysis (`--nextpnr` / `--pnr-only`)**:
   Runs Yosys and nextpnr, computing exact physical device utilization and maximum operating frequency ($F_{\max}$), without generating the final `.fs` bitstream file.
   ```bash
   python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k --nextpnr
   ```

### Board Profiles & Physical Pin Constraints

- **Board Profiles (`boards/*.json`)**: Hardware targets (such as `boards/tang-primer-20k.json`) configure the EDA synthesis script, placement arguments, bitstream packer, and physical capacity limits (LUT4, FF, BRAM, DSP).
- **Physical Pin Constraints (`boards/constraints/*.cst`)**: Pin mappings (clock, reset, UART RX/TX) are automatically adapted to the top module's exact port names (`clk`, `reset_n`/`io_resetN`, `uart_rx`/`io_uartRx`, `uart_tx`/`io_uartTx`). You can also override the constraint file via `--cst <path.cst>`.

Options:
- `src` : Source `.scala` model file, Verilog directory, or single `.v` file [default: `rtl/`].
- `-o`, `--out <PATH>` : Output directory for build artifacts [default: `hw_build/<board>/`].
- `--board <NAME>` : Target FPGA board profile [default: `tang-primer-20k`].
- `--cst`, `--constraints <PATH>` : Custom physical pin constraints file override (`.cst`).
- `--top <NAME>` : Top-level module name (auto-detected if omitted: `UartSoC`, `top`).
- `--synth-only`, `--yosys` : Stop after Yosys synthesis.
- `--pnr-only`, `--nextpnr` : Stop after nextpnr place & route.
- `--clk <FREQ>` : Target clock frequency override (e.g. `'27MHz'`, `'50MHz'`).

---

## 4. FPGA Hardware Flashing & Deployment (`flash`)

The `flash` command programs the target FPGA hardware using `openFPGALoader`. It automatically resolves the bitstream from `hw_build/<board>/<bitstream_name>` if omitted.

### SRAM Volatile Programming

Fast loading (~1.5 seconds) directly into FPGA volatile SRAM. Ideal for testing, verification, and active development:

```bash
python cli/main.py flash --board tang-primer-20k
```

### SPI Flash Non-Volatile Programming

Permanently burns the bitstream into the board's on-board SPI Flash memory. The circuit remains configured even after power cycling:

```bash
python cli/main.py flash --board tang-primer-20k --flash
```

Options:
- `bitstream` : Path to `.fs` / `.bit` bitstream file (auto-detected in `hw_build/<board>/` if omitted).
- `--board <NAME>` : Target FPGA board profile [default: `tang-primer-20k`].
- `--sram / --no-sram` : Load into volatile SRAM [default: `--sram`].
- `--flash` : Program on-board non-volatile SPI Flash memory.

---

## 5. Circuit Simulation & Verification (`test`)

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

## 6. Full Regression Testing (`test-all`)

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

## 7. Formal Verification Engine (`test-all-formal`)

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

## 8. Python Hardware Co-Simulations (`test-all-python`)

SpinalML features end-to-end Python/Cocotb hardware co-simulations to test neural network layers and operators directly against Python golden models (Torch/NumPy).

> [!NOTE]
> **Platform Requirements**: Cocotb's official VPI bridge architecture for Verilator requires a POSIX environment (`libcocotbvpi_verilator.so` and `-ldl`).
> * **Linux / ARM64 / Radxa Rock**: Supported natively.
> * **Windows**: Supported via **WSL (Windows Subsystem for Linux)** (`wsl python cli/main.py test-all-python`).
> *(Note: ScalaTest dynamic simulations `test-all` and Formal Verification `test-all-formal` run 100% natively on both Windows and Linux without WSL!)*

`test-all-python` automatically injects `VERILATOR_ROOT`, `oss-cad-suite/bin`, and `mill` into the simulation environment so that tests run out-of-the-box:

```bash
# Linux or inside WSL:
python cli/main.py test-all-python

# From Windows PowerShell via WSL:
wsl python cli/main.py test-all-python

# Or with verbose Cocotb output
python cli/main.py test-all-python -v
```

### Options & Filtering

* **Filter tests by name pattern** (`-k`, `--filter`):
  ```bash
  python cli/main.py test-all-python -k "matmul"
  ```
* **Stop on first failure** (`-x`, `--fail-fast`):
  ```bash
  python cli/main.py test-all-python --fail-fast
  ```
* **Debug true mathematical precision** (`--debug-math`):
  Generates `true_math_errors.log` comparing hardware bits against high-precision float golden models:
  ```bash
  python cli/main.py test-all-python --debug-math
  ```

---

## 9. Low-Level EDA Tool Passthroughs

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

## 10. Quick Command Reference

| Action | Linux Command | Windows PowerShell Command |
| :--- | :--- | :--- |
| **Install `uv`** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Create venv** | `uv venv -p 3.11` | `uv venv -p 3.11` |
| **Install requirements** | `uv pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| **Activate venv** | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| **Install tools** | `python cli/main.py setup` | `python cli/main.py setup` |
| **Compile to Verilog** | `python cli/main.py compile spinalML/src/spinalML/examples/Mnist.scala -o verilog/` | `python cli/main.py compile spinalML\src\spinalML\examples\Mnist.scala -o verilog\` |
| **Build FPGA Bitstream** | `python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k` | `python cli/main.py build tests\universal\Universal1DDemo.scala --board tang-primer-20k` |
| **Build (Yosys Synth Only)** | `python cli/main.py build tests/universal/Universal1DDemo.scala --yosys` | `python cli/main.py build tests\universal\Universal1DDemo.scala --yosys` |
| **Build (nextpnr PnR Only)** | `python cli/main.py build tests/universal/Universal1DDemo.scala --nextpnr` | `python cli/main.py build tests\universal\Universal1DDemo.scala --nextpnr` |
| **Flash FPGA (SRAM)** | `python cli/main.py flash --board tang-primer-20k` | `python cli/main.py flash --board tang-primer-20k` |
| **Flash FPGA (SPI Flash)** | `python cli/main.py flash --board tang-primer-20k --flash` | `python cli/main.py flash --board tang-primer-20k --flash` |
| **Universal Circuit Test** | `python cli/main.py test spinalML/src/spinalML/examples/Mnistw4a8.scala` | `python cli/main.py test spinalML\src\spinalML\examples\Mnistw4a8.scala` |
| **Single ScalaTest** | `python cli/main.py test spinalML/test/src/spinalML/examples/MnistTest.scala` | `python cli/main.py test spinalML\test\src\spinalML\examples\MnistTest.scala` |
| **Run All Dynamic Tests** | `python cli/main.py test-all` | `python cli/main.py test-all` |
| **Run Filtered Tests** | `python cli/main.py test-all -k "Conv2D"` | `python cli/main.py test-all -k "Conv2D"` |
| **Run All Formal Proofs** | `python cli/main.py test-all-formal` | `python cli/main.py test-all-formal` |
| **Run Filtered Formal** | `python cli/main.py test-all-formal -k "Accelerator"` | `python cli/main.py test-all-formal -k "Accelerator"` |
| **Direct Mill Command** | `python cli/main.py mill spinalML.compile` | `python cli/main.py mill spinalML.compile` |
| **Clean Coursier Cache** | `python cli/main.py clean-cache` | `python cli/main.py clean-cache` |
