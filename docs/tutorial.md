# SpinalML Quick Tutorial

Welcome to the **SpinalML** quick tutorial. This guide explains how to compile, simulate, and generate Verilog for your first machine learning hardware component using the integrated **SpinalML CLI**.

---

## 1. Prerequisites & Setup

SpinalML automates its entire environment provisioning (Mill, Verilator, Yosys, nextpnr). You only need Python and [`uv`](https://docs.astral.sh/uv/):

```bash
# Setup Python environment
uv venv -p 3.11
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt

# Provision all hardware tools (Mill, Verilator, etc.)
python cli/main.py setup
```

---

## 2. Running Hardware Simulations

To verify your setup, run cycle-accurate simulations compiled with Verilator via the CLI:

```bash
# Run a specific model simulation
python cli/main.py test tests/universal/Universal1DDemo.scala

# Or run the full ScalaTest suite
python cli/main.py test-all
```

The CLI automatically:
1. Resolves all SpinalHDL dependencies via Mill.
2. Elaborates the Scala model into synthesizable Verilog.
3. Compiles the testbench and Verilog model with Verilator into a cycle-accurate C++ binary.
4. Executes test stimulus and verifies bit-exact tensor outputs.

---

## 3. Generating Standalone Verilog RTL

You **never** need to write manual `object ... extends App { SpinalVerilog(...) }` boilerplate runners. The CLI automatically discovers and synthesizes your component or accelerator:

```bash
# Emit standalone Verilog into the verilog/ directory
python cli/main.py compile tests/universal/Universal1DDemo.scala -o verilog/
```

This generates the complete synthesizable RTL netlist (`.v`) ready for FPGA synthesis or ASIC tooling.
