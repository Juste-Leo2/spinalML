# Project Structure & Architecture

This document maps the **SpinalML** repository: architectural partitioning, codebase organization, directory layouts, and the verification framework.

SpinalML is an open-source hardware Machine Learning acceleration library written in Scala using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/).

---

## 1. Dual-Target Architecture: The 3-Layer Sandwich

To serve both as a bit-exact FPGA emulator and as standard-cell RTL ready for ASIC tapeout, SpinalML partitions hardware responsibilities into three decoupled layers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: SPINALML ALGORITHMIC CORE (100% Portable RTL)                │
│  - Valid/ready streaming handshakes (`Stream[Tensor]`), backpressure   │
│  - Generic arithmetic operators (Conv2D, Matmul, Activations)          │
│  - Universal SIMD datapath (configurable `lanes: Int`)                 │
│  - Target trait (`Target.FPGA` / `Target.ASIC`)                        │
│  - ZERO vendor primitives (no hardcoded Gowin/Xilinx/Altera macros)    │
│  - ZERO hardcoded memory addresses or physical mappings                │
├────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: ABSTRACT MEMORY ADAPTER (`MemoryAdapter`)                    │
│  - Abstract memory interface feeding Layer 1                           │
│  - Swappable implementations according to target:                      │
│    * `BramAdapter`     : On-chip BRAM for small FPGAs / simulation     │
│    * `DdrAdapter`      : AXI / DMA controller for larger FPGAs         │
│    * `SramAsicAdapter` : OpenRAM SRAM macros for ASIC (Sky130, etc.)   │
├────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: PHYSICAL & BOARD INTEGRATION (`SoCTop` / `BoardTop`)         │
│  - Physical constraints (.cst Gowin, .xdc Xilinx, .sdc ASIC)           │
│  - Hardware reset strategy (synchronous vs asynchronous)               │
│  - Physical UART bridge / host communication                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Directory Layout

```text
spinalML/
├── build.mill                  # Mill build configuration (Scala 2.12, SpinalHDL 1.15.0)
├── README.md                   # Project overview, quickstart, and documentation index
├── CONTRIBUTING.md             # Contribution guidelines & development conventions
├── LICENSE                     # MIT License
├── CITATION.cff                # Citation metadata
├── requirements.txt           # CLI runtime only (typer, rich, requests); full envs are uv-managed
├── requirements/              # Managed env pins for `spinalml setup` (base, dram, dev)
│
├── boards/                     # Hardware target definitions
│   ├── tang-primer-20k.json    # Sipeed Tang Primer 20K (Gowin GW2A-18) configuration
│   └── constraints/
│       └── tang-primer-20k.cst # Physical pin constraints (Clock, UART, LEDs)
│
├── cli/                        # SpinalML unified CLI toolchain
│   ├── main.py                 # CLI entry point (`python cli/main.py`)
│   ├── config.json             # Toolchain paths & default configuration
│   └── spinalml_cli/           # CLI execution modules
│       ├── build_runner.py     # Yosys -> nextpnr -> gowin_pack pipeline
│       ├── flash_runner.py     # openFPGALoader integration
│       ├── test_runner.py      # Mill / Verilator ScalaTest runner
│       ├── installer.py        # Automated toolchain downloader & setup
│       └── uart_host.py        # Host UART communication client
│
├── docs/                       # Technical documentation
│   ├── getting_started.md      # Beginner guide: tensors, streams, basic layers
│   ├── HighLevelTutorial.md    # PyTorch-like API, quantization, and DAG topologies
│   ├── opsDocs.md              # Operations and LayerSpec API reference
│   ├── cli.md                  # CLI commands manual
│   ├── uart_bridge.md          # UART CSR protocol and packet framing
│   ├── project_structure.md    # Repository layout and architecture (this file)
│   ├── roadmap.md              # High-level strategic roadmap & milestones
│   ├── full_roadmap.md         # Exhaustive engineering backlog and tracking
│   ├── roadmap_board.md        # Hardware board compatibility matrix
│   └── bugs/                   # Hardware post-mortems and bug reports
│
├── examples/                   # End-to-end applications
│   └── Mnist/                  # Mixed-precision W4A8 CNN on Tang Primer 20K
│       ├── Model.scala         # Hardware accelerator description
│       ├── inference.py        # Gradio web canvas & real-time UART client
│       ├── Mnist_weights.npz   # Trained quantized model weights
│       └── top.fs              # Pre-compiled bitstream
│
├── scripts/                    # Hardware host utilities
│   └── test_hardware_1d.py     # Host script for UART physical FPGA inference
│
├── spinalML/
│   ├── src/spinalML/           # Hardware RTL source code
│   │   ├── dtypes/             # Numerical types (FloatML, BF16, FP8, I4, I8, I16, I32)
│   │   ├── dsp/                # Hardware arithmetic & Target trait (FPGA / ASIC)
│   │   ├── tensors/            # Tensor[T] abstraction and streaming interfaces
│   │   ├── interfaces/         # AXI4, AXI4-Lite, and AXI4-Stream converters
│   │   ├── ops/                # Primitive mathematical and tensor operations
│   │   │   ├── add.scala, mul.scala, sub.scala, div.scala
│   │   │   ├── matmul.scala, dot.scala, scale_add.scala
│   │   │   ├── exp.scala, sqrt.scala, rsqrt.scala, cast.scala
│   │   │   ├── reshape.scala, flatten.scala, repack.scala, transpose.scala
│   │   │   └── seq2col.scala, im2col.scala
│   │   ├── activations/        # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   │   ├── poolings/           # MaxPool1D, AvgPool1D, MaxPool2D, AvgPool2D
│   │   ├── layers/             # Conv1D, Conv2D, Linear, BatchNorm, LayerNorm
│   │   ├── nn/                 # High-level declarative API (Sequential, Accelerator)
│   │   ├── memory/             # DMA engines (DMAReader, DMAWriter, LineBuffer2D, StreamDoubleBuffer)
│   │   └── examples/           # Standalone Scala RTL generation templates
│   │
│   └── test/src/spinalML/      # Testbenches & verification
│       ├── ops/, layers/, ...  # Unit tests mirroring src/
│       ├── harness/            # Universal SoC simulation harness & memory models
│       ├── replica/            # Bit-exact software reference oracles
│       └── symbolicTest/       # Formal verification specs (SymbiYosys + CVC4/Z3)
│
└── tests/
    ├── universal/              # Canonical hardware models for FPGA deployment
    └── python/                 # Cocotb + Verilator co-simulations with NumPy golden models
```

---

## 3. Multi-Level Verification Strategy

SpinalML combines three independent verification layers to guarantee correctness from mathematical invariants to silicon execution:

1. **Formal Verification (SymbiYosys / BMC)**:
   - Exhaustively proves control flows, stream handshakes (`valid` / `ready`), and absence of deadlocks across all modules and operations using SMT solvers (`cvc4`).
   - Proves proper backpressure propagation and protocol invariants under all possible timing sequences.
   - Run via: `python cli/main.py test-all-formal`

2. **Cycle-Accurate Simulation (Verilator 5)**:
   - High-speed C++ simulation compiling the complete generated Verilog RTL with AXI4 memory models.
   - Executed through Mill test suites: `python cli/main.py test-all`

3. **Universal Test Engine & Bit-Exact Software Oracles (`ModelReplica` & Cocotb)**:
   - The CLI test command (`python cli/main.py test <model.scala>`) automatically generates an SoC simulation harness, loads deterministic weights, streams inputs, and verifies that hardware output matches the software oracle with zero numerical deviation (`deviation = 0.000`).
   - Python co-simulations cross-check mathematical operations against NumPy references: `python cli/main.py test-all-python`