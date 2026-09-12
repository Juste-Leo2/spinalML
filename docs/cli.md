# SpinalML Command Line Interface (CLI)

The **SpinalML CLI** (`cli/main.py`) provides an integrated, cross-platform toolchain management and verification suite for developing, compiling, and deploying machine learning hardware accelerators in SpinalHDL.

It manages external EDA dependencies (Mill, Verilator, SymbiYosys, Yosys, nextpnr, openFPGALoader) with zero manual environment pollution.

---

## Fast-Track: 4 Essential Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **1. Toolchain Setup** | `python cli/main.py setup` | Provisions Mill, Verilator, Yosys, nextpnr, and openFPGALoader automatically |
| **2. Dynamic Simulation** | `python cli/main.py test <target.scala>` | Runs cycle-accurate C++ simulation via Mill & Verilator |
| **3. Turnkey FPGA Build** | `python cli/main.py build <target.scala> --board tang-primer-20k` | Generates Verilog, wraps UartSoC, synthesizes with hardware DSPs, and outputs `.fs` bitstream |
| **4. Hardware Flash** | `python cli/main.py flash --board tang-primer-20k` | Flashes the built bitstream to FPGA SRAM in ~1.5s via openFPGALoader |

> [!NOTE]
> Hardware DSP blocks (`MULT18X18` with output pipeline registers) are enabled by default across all supported FPGA targets, saving up to ~30% of logic LUTs while achieving reliable, numerically validated hardware inference on physical silicon (resolving the combinational Gowin sign bug, see [DSP Resolution Note](bugs/2026-09-gowin-dsp-combinational-signedness.md)).
> The `--no-dsp` flag remains available as an opt-in fallback to force all arithmetic into soft LUT logic if desired.

---

## 1. Environment Setup

SpinalML uses [**uv**](https://docs.astral.sh/uv/) for high-speed Python dependency management.

### Installation
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Virtual Environment & Dependencies
```bash
# Clone the repository
git clone https://github.com/Juste-Leo2/spinalML.git
cd spinalML

# Create virtualenv and install dependencies
uv venv -p 3.12.1
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt

# Provision EDA tools (Mill, Verilator, Yosys, nextpnr, openFPGALoader)
python cli/main.py setup
```

---

## 2. Hardware Compilation (`compile`)

Elaborates SpinalHDL models and emits standalone synthesizable Verilog (`.v`).

```bash
python cli/main.py compile <path_to_scala_file> -o verilog/
```

- If your file defines an `Accelerator` or `Component` class without a main object, the CLI auto-generates an ephemeral runner to synthesize it.
- **Key options**:
  - `-o, --out <DIR>` : Destination directory for emitted Verilog.
  - `--debug` : Show full Mill elaboration stack traces.

---

## 3. Turnkey FPGA Synthesis & Packing (`build`)

Converts a Scala model directly into a ready-to-flash bitstream (`hw_build/<board>/top.fs`):

```bash
python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k --no-dsp
```

### Build Pipeline:
1. **Elaboration**: Compiles Scala to Verilog, wrapping your model inside the turnkey `UartSoC` (AXI4 Master, CSR registers, BRAM, UART Bridge).
2. **Synthesis (Yosys)**: Maps RTL to target device primitives (LUTs, FFs, BRAMs).
3. **Place & Route (nextpnr)**: Places and routes logic, calculating $F_{\max}$ and timing closure.
4. **Bitstream Packing (gowin_pack)**: Assembles the bitstream ready for SRAM/Flash.

### Useful Build Options:
- `--board <NAME>` : Board profile (default: `tang-primer-20k`, loaded from `boards/`).
- `--no-dsp` : Disables hard DSP inference in Yosys (forces LUT implementation).
- `--synth-only` / `--yosys` : Stops after Yosys to quickly inspect cell and resource counts.
- `--pnr-only` / `--nextpnr` : Stops after nextpnr place & route (without bitstream generation).
- `--clk <FREQ>` : Overrides target clock frequency (e.g. `'27MHz'`, `'50MHz'`).

---

## 4. FPGA Hardware Flashing (`flash`)

Programs the target FPGA using `openFPGALoader`. The CLI automatically discovers the latest bitstream in `hw_build/<board>/`.

### Fast SRAM Programming (Volatile)
```bash
python cli/main.py flash --board tang-primer-20k
```

### SPI Flash Programming (Non-Volatile)
```bash
python cli/main.py flash --board tang-primer-20k --flash
```

---

## 5. Circuit Simulation & Verification (`test`)

Executes cycle-accurate simulation tests using Mill and Verilator:

```bash
# Run a specific Scala test file or model
python cli/main.py test tests/universal/Universal1DDemo.scala

# Enable VCD waveform dumping (saved to simWorkspace/)
python cli/main.py test tests/universal/Universal1DDemo.scala --wave
```

---

## 6. Regression & Full Test Suites

SpinalML includes complete verification across three distinct tiers:

```bash
# 1. Run all ScalaTest unit simulations
python cli/main.py test-all

# 2. Run formal verification proofs (SymbiYosys + Z3 k-induction)
python cli/main.py test-all-formal

# 3. Run Python/Cocotb co-simulations against NumPy golden models
python cli/main.py test-all-python
```

---

## 7. Physical Hardware Inference (`scripts/test_hardware_1d.py`)

After flashing a universal demo bitstream to the board, run real-time physical inference over the USB-UART bridge:

```bash
python scripts/test_hardware_1d.py --port COM8
# On Linux:
python scripts/test_hardware_1d.py --port /dev/ttyUSB0
```

The script streams test vectors, monitors CSR handshakes, reads output activations, and validates bit-exact equality against the software model.
