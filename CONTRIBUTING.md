# Contributing to SpinalML

Thank you for your interest in contributing to **SpinalML**! 

Whether you are an FPGA designer, machine learning researcher, embedded software developer, or open-source enthusiast, your contributions are warmly welcomed.

---

## 1. Where You Can Help Most

Here are several high-impact areas where community contributions make a tremendous difference:

### A. Porting & Validating New FPGA Boards
Physical in-circuit testing is currently performed primarily on the **Sipeed Tang Primer 20K** because it is the main physical board available to the author. 

If you own other FPGA hardware, your help in adding and validating board targets is hugely appreciated:
- **AMD / Xilinx**: Digilent Arty A7 (35T / 100T), Basys 3, PYNQ-Z2, Cora Z7, Kria KV260/KR260.
- **Gowin**: Sipeed Tang Nano 9K / 20K, Tang Mega 138K.
- **Lattice**: iCEBreaker (iCE40UP5K), Colorlight 5A-75B / OrangeCrab (ECP5).
- **Intel / Altera**: Terasic DE10-Lite (MAX 10), DE10-Nano (Cyclone V).

You can help validate bitstreams, test UART communication, or submit a ready-to-use board profile in [`boards/`](boards/)!

### B. Hardware Neural Network Operators & Layers
- Adding new activation functions (e.g. GELU, SiLU, LeakyReLU).
- Implementing new vision and sequence operations (Dilated Conv, Depthwise-Separable Conv, LSTM / GRU).
- Optimizing line buffers and streaming accumulators.

### C. Quantization & Math Hardware Primitives
- Hard DSP inference mapping (pipelined MAC units for Gowin, Xilinx DSP48E1/E2, ECP5 MULT18).
- Sub-byte quantization primitives (INT2, INT4, FP4).
- Softmax / LayerNorm hardware approximation blocks.

### D. Toolchain & CLI Improvements
- Enhancing board detection and programmer interfaces (`openFPGALoader`, `vivado`, `quartus`).
- Improving timing and resource utilization parsers.

### E. Documentation, Tutorials & Real-World Demos
- Writing guides, improving docstrings, fixing typos.
- Developing end-to-end example projects in [`examples/`](examples/).

---

## 2. Development Setup

Follow the quick setup guide to configure your Python environment and open-source EDA toolchain:

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/spinalML.git
cd spinalML

# 2. Setup virtual environment with uv
uv venv -p 3.12.1
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Provision the hardware EDA toolchain (Mill, Verilator, Yosys, nextpnr)
python cli/main.py setup
```

---

## 3. How to Add a New FPGA Board Target

Adding a new board profile is modular and requires no modifications to the SpinalHDL core:

1. **Create a JSON profile** in `boards/<board-slug>.json`:
   - Specify clock frequency, baud rate, default BRAM words, synthesis command, PnR arguments, and programmer target (e.g. `openFPGALoader`).
   - Define hardware resource capacities (`lut`, `ff`, `bram`, `dsp`) for automated utilization reporting.
2. **Add pin constraints** in `boards/constraints/<board-slug>.<cst|xdc|lpf>`:
   - Map top-level ports (`clk`, `reset_n`, `uart_rx`, `uart_tx`, etc.) to board package pins.
3. **Test synthesis & bitstream generation**:
   ```bash
   python cli/main.py build tests/universal/Universal1DDemo.scala --board <board-slug> --no-dsp
   ```
4. **Flash and verify on physical hardware**:
   ```bash
   python cli/main.py flash hw_build/<board-slug>/top.fs --board <board-slug>
   ```

---

## 4. Verification & Testing Workflow

Hardware changes must always be verified before submission:

```bash
# Run ScalaTest unit and integration tests
python cli/main.py test-all

# Run Python / Cocotb / Verilator co-simulations
python cli/main.py test-all-python

# Run formal verification (SymbiYosys) if applicable
python cli/main.py test-all-formal
```

Ensure all existing tests pass cleanly in your local environment.

---

## 5. Pull Request Guidelines

When submitting a Pull Request:
1. **Focused Scope**: Keep changes focused on a single bug fix, feature, or board target profile per PR.
2. **Descriptive Summary**: Explain what was changed, the design choices made, and the hardware implications (timing, resource utilization, latency).
3. **Proof of Verification**: Attach simulation logs (`test-all`), synthesis reports (`build`), or hardware in-circuit test outputs to your PR description.

---

## 6. Using AI Coding Assistants

AI tools (Copilot, Claude, ChatGPT, Gemini, etc.) are a natural part of modern workflows, and you are welcome to use them to assist your development.

At the end of the day, **what matters most is the code itself: that it is clean, well-integrated, and actually works**:
- **No need to be an FPGA guru**: You don't have to be a hardware veteran to contribute! We only ask that you understand the broad strokes of what your changes do and can explain them simply.
- **Human PR descriptions**: Please write the PR description yourself in your own words, explaining what you are adding or fixing.
- **Verify before submitting**: Make sure the code compiles and passes local tests before opening a PR. For example, if you are adding support for a new board or layer, run the build/simulation and attach the logs so everyone can see it working end-to-end.

---

Thank you for helping push the frontier of open-source hardware machine learning with SpinalHDL!
