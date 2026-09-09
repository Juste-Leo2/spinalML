# Project Structure

This document maps the **SpinalML** repository: where components live, how they interact, and how the codebase is organized for contributors.

SpinalML is a hardware Machine Learning acceleration library written in Scala with [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/).

---

## Repository Overview

- **`spinalML/src/`** — Core hardware library (Scala / SpinalHDL): data types, tensor streams, primitive operations, neural network layers, memory engines, and RTL examples.
- **`spinalML/test/`** — ScalaTest simulation testbenches mirroring `src/`, plus symbolic (formal) verification specs in `symbolicTest/`.
- **`cli/`** — Unified Python CLI (`cli/main.py`) automating toolchain management, simulation, synthesis, placement, routing, and flashing.
- **`boards/`** — FPGA board profiles (JSON) and pin constraint files (`.cst`).
- **`tests/`** — Universal hardware test engines (`tests/universal/`) and Python/Cocotb/Verilator bit-exact co-simulations (`tests/python/`).
- **`scripts/`** — Host driver scripts (e.g. UART physical inference client `test_hardware_1d.py`).
- **`examples/`** — End-to-end applications pairing Python ML training with FPGA deployments (e.g. upcoming `examples/mnist/`).
- **`docs/`** — Complete technical documentation, guides, and hardware post-mortems (`docs/bugs/`).

---

## File Tree

```text
spinalML/
├── build.mill                  # Mill build configuration (Scala 2.12, SpinalHDL 1.14.2)
├── README.md                   # Project overview, quickstart, and documentation index
├── CONTRIBUTING.md             # Contribution rules & Generative AI policy
├── LICENSE                     # MIT License
├── CITATION.cff                # Citation metadata
├── requirements.txt           # Python dependencies (typer, rich, cocotb, numpy, pytest)
│
├── boards/                     # Hardware target definitions
│   ├── tang-primer-20k.json    # Sipeed Tang Primer 20K (Gowin GW2A-18) configuration
│   └── constraints/
│       └── tang-primer-20k.cst # Physical pin constraints (Clock, UART, LEDs)
│
├── cli/                        # SpinalML unified CLI toolchain
│   ├── main.py                 # CLI entry point
│   ├── config.json             # Toolchain paths & configuration
│   └── spinalml_cli/           # CLI execution modules
│       ├── build_runner.py     # Yosys -> nextpnr -> gowin_pack pipeline
│       ├── flash_runner.py     # openFPGALoader integration
│       ├── test_runner.py      # Mill / Verilator ScalaTest runner
│       ├── installer.py        # Toolchain auto-downloader & setup
│       └── uart_host.py        # Host UART communication helper
│
├── docs/                       # Documentation
│   ├── getting_started.md      # Beginner guide: tensors, streams, basic layers
│   ├── HighLevelTutorial.md    # PyTorch-like API, quantization, and DAG topologies
│   ├── opsDocs.md              # Concise hardware operations reference
│   ├── opsSupport.md           # Support matrix and validation status
│   ├── cli.md                  # CLI usage and commands
│   ├── uart_bridge.md          # UART CSR protocol and packet framing
│   ├── project_structure.md    # Repository layout (this file)
│   └── bugs/                   # Hardware post-mortems & bug reports
│       └── 2026-09-gowin-dsp-combinational-signedness.md
│
├── examples/                   # End-to-end ML applications
│   └── README.md               # Application catalog and workflow guide
│
├── scripts/                    # Hardware host utilities
│   └── test_hardware_1d.py     # Host script for UART physical FPGA inference
│
├── spinalML/
│   ├── src/spinalML/           # Hardware RTL source code
│   │   ├── dtypes/             # Numerical types (FloatML, BF16, FP8, FP4, I8, I16, I32)
│   │   ├── tensors/            # Tensor[T] abstraction and streaming interfaces
│   │   ├── interfaces/         # AXI4-Stream converters
│   │   ├── ops/                # Primitive mathematical and tensor operations
│   │   │   ├── add.scala, mul.scala, sub.scala, div.scala
│   │   │   ├── matmul.scala, dot.scala, scale_add.scala
│   │   │   ├── exp.scala, sqrt.scala, rsqrt.scala, reciprocal.scala
│   │   │   ├── reshape.scala, flatten.scala, repack.scala, transpose.scala
│   │   │   └── seq2col.scala, im2col.scala
│   │   ├── activations/        # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   │   ├── poolings/           # MaxPool1D, AvgPool1D, MaxPool2D, AvgPool2D
│   │   ├── layers/             # Conv1D, Conv2D, Linear, BatchNorm, LayerNorm
│   │   ├── nn/                 # High-level declarative API (Sequential, Accelerator)
│   │   ├── memory/             # AXI4 DMAReader, DMAReader2D, StreamDoubleBuffer
│   │   └── examples/           # Standalone Scala RTL generation templates
│   │
│   └── test/src/spinalML/      # Simulation testbenches
│       ├── ops/, layers/, ...  # Unit tests mirroring src/
│       └── symbolicTest/       # Formal verification specs (SymbiYosys + Z3)
│
└── tests/
    ├── universal/              # Universal test models for FPGA bitstream deployment
    │   └── Universal1DDemo.scala
    └── python/                 # Cocotb + Verilator co-simulations with NumPy golden models
```

---

## Conventions & Verification Strategy

- **Scala Simulation (`test`)**: Run dynamic tests with Mill and Verilator (`python cli/main.py test <target>`).
- **Python Golden Models (`test-all-python`)**: Every mathematical operation is verified against a bit-exact NumPy reference model.
- **Formal Verification (`test-all-formal`)**: SymbiYosys proves arithmetic correctness and absence of deadlocks via k-induction.
- **Physical Bitstream Build (`build`)**: Yosys synthesizes RTL, nextpnr places & routes, and vendor tools generate the final bitstream for FPGA deployment.