# SpinalML CLI

The **SpinalML** command-line interface (CLI) makes it easy to orchestrate the entire FPGA design flow (SpinalHDL compilation, Verilator simulations, Yosys synthesis, NextPNR place-and-route, and openFPGALoader hardware programming) without requiring complex system-wide installations.

> [!IMPORTANT]
> **Golden rule:** Whether you run from Python sources (`python cli/main.py`) or the standalone binary (`./spinalml`), commands must **always be executed from the project root** (`spinalML/`).

---

## 1. Precompiled Standalone Binaries (Recommended)

Single-file standalone executables are automatically built and available for download on the [**GitHub Releases**](https://github.com/Juste-Leo2/spinalML/releases) page:

| Platform | Release Archive | Executable Binary |
| :--- | :--- | :--- |
| **Windows (x64)** | `spinalml-windows-x64.zip` | `spinalml-windows-x64.exe` |
| **Linux (x64)** | `spinalml-linux-x64.tar.gz` | `spinalml-linux-x64` |
| **Linux (ARM64)** | `spinalml-linux-arm64.tar.gz` | `spinalml-linux-arm64` |
| **macOS (Apple Silicon)** | `spinalml-darwin-arm64.tar.gz` | `spinalml-darwin-arm64` |
| **macOS (Intel)** | `spinalml-darwin-x64.tar.gz` | `spinalml-darwin-x64` |

### What's inside the binary?
- **Ultra-lightweight (~15 MB)**: The binary bundles the pure CLI engine (`typer`, `rich`, `requests`), FPGA board profiles (`boards/`), the hardware framework (`spinalML/`), and the build configuration (`build.mill`).
- **What is NOT bundled**: Python co-simulation tests (`cocotb`, `pytest`). Cocotb requires dynamic VPI compilation linked against the local system's CPython 3.12 (on Linux or WSL). Users who want to run the development Python test suite should use the full source environment (see [`docs/building_from_source.md`](../docs/building_from_source.md)).

### Quickstart with the binary:
```bash
# 1. Download and extract the executable for your OS from Releases
# 2. Navigate to the root of your SpinalML project
cd spinalML

# 3. Initialize EDA tools (OSS CAD Suite, Mill, w64devkit)
./spinalml setup

# 4. Compile a Scala model into Verilog
./spinalml compile tests/universal/UniversalOpsDemo.scala -o rtl/

# 5. Run bit-exact verification under Verilator
./spinalml test tests/universal/UniversalOpsDemo.scala
```

---

## 2. Building Binaries from Source

To build the standalone binary yourself with PyInstaller:

```bash
# From the spinalML project root:
uv pip install -r requirements.txt pyinstaller

# Run the build script
python cli/build_binary.py
```
The resulting binary will be generated in the `dist/` directory.

---

## 3. Running from Python Sources

If you prefer running the CLI without building a binary:

1. **Install the minimal dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
2. **Invoke the CLI via `python cli/main.py`:**
   ```bash
   python cli/main.py --help
   ```

---

## 4. Main Commands

| Command | Description |
| :--- | :--- |
| `spinalml setup` | Automatically downloads and configures Mill and OSS CAD Suite in `~/.spinalml_tools` |
| `spinalml compile <model.scala>` | Compiles a Scala model into standalone Verilog and generates the UartSoC wrapper |
| `spinalml test <model.scala>` | Runs universal bit-exact verification under Verilator C++ simulation |
| `spinalml build <model.scala>` | Yosys synthesis -> NextPNR place & route -> `.fs` bitstream generation |
| `spinalml flash` | Programs the connected USB FPGA board via openFPGALoader |
| `spinalml mill [...]` | Direct wrapper for the Mill Scala build tool |
| `spinalml verilator [...]` | Direct wrapper for the Verilator simulator |
| `spinalml yosys [...]` | Direct wrapper for the Yosys synthesis tool |
| `spinalml nextpnr <arch> [...]` | Direct wrapper for NextPNR (`ice40`, `himbaechel`, etc.) |
| `spinalml openfpgaloader [...]` | Direct wrapper for the JTAG/UART programming utility |
