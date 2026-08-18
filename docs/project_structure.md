# Project Structure

This document maps the spinalML repository: where things live and what they are for. It is written for contributors who want to quickly understand the layout.

spinalML is a Machine Learning library for hardware synthesis and simulation, written in Scala with [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/). The main entry point for users is the [README](../README.md).

## Repository overview

- **`spinalML/src/`** — the hardware library itself (Scala/SpinalHDL) and its examples.
- **`spinalML/test/`** — ScalaTest unit tests, mirroring the layout of `src/`, plus the symbolic (formal) specifications in `symbolicTest/`.
- **`tests/python/`** — Python co-simulation (Cocotb + Verilator) validating hardware models against NumPy golden models.
- **`docs/`** — all project documentation.

Everything not listed below (`out/`, `sim_build/`, `simWorkspace/`, `rtl/`, `.venv/`, `*.v` at the root) is gitignored build output; none of it is part of the source tree.

## File tree

```
spinalML/
├── build.sc                  # Mill build: Scala 2.12, SpinalHDL 1.14.2, ScalaTest
├── README.md                 # Overview, docs index, contribution policy
├── LICENSE                   # MIT
├── requirements.txt          # Python deps: pytest, cocotb, cocotb-test, numpy, cocotbext-axi
├── .gitignore
├── mill                      # self-contained Mill launcher (not tracked)
├── .github/workflows/
│   ├── ci-simulations.yml    # runs the hardware simulations (Verilator) on push to main
│   ├── ci-python.yml         # runs the Python/Cocotb co-sim, after the HW sims pass
│   ├── ci-symbolic.yml       # symbolic (formal) proofs: Yosys + SymbiYosys + Z3
│   └── ci-sentrux.yml        # code-health checks (Sentrux)
├── .sentrux/
│   └── rules.toml            # no cyclic deps, no god files, max cyclomatic complexity,
│                             #   and src must never depend on test code
├── docs/
│   ├── getting_started.md    # tutorial: tensors, streams, building your first ML layers
│   ├── tutorial.md           # follow-up tutorial material
│   ├── opsDocs.md            # API reference of all hardware operations
│   ├── opsSupport.md         # ops x hardware-validation-status matrix
│   ├── pythonTest.md         # docs of the Python/Cocotb/Verilator test framework
│   ├── symbolicTest.md       # symbolic (formal) testing: install, architecture, results
│   ├── symbolicTestPlaybook.md # hands-on guide to write any formal spec (templates, API)
│   ├── roadmap.md            # development phases checklist
│   └── scratch/              # WIP scratch space (empty)
├── spinalML/
│   ├── src/spinalML/
│   │   ├── dtypes/           # numerical formats
│   │   │   ├── FloatML.scala     # generic float parametrized by exp/mant bits
│   │   │   ├── BF16.scala        # BFloat16
│   │   │   ├── FP4.scala         # FP4 E2M1
│   │   │   ├── FP8.scala         # FP8 E4M3
│   │   │   ├── I4.scala          # 4-bit signed int
│   │   │   ├── I8.scala          # 8-bit signed int
│   │   │   ├── I16.scala         # 16-bit signed int
│   │   │   ├── I32.scala         # 32-bit signed int
│   │   │   ├── U4.scala          # 4-bit unsigned int
│   │   │   └── U8.scala          # 8-bit unsigned int
│   │   ├── tensors/
│   │   │   └── Tensor.scala      # tensor representation + dataflow streams
│   │   ├── interfaces/
│   │   │   └── Axi4StreamConverter.scala  # tensor streams <-> AXI4-Stream
│   │   ├── ops/               # primitive operations (names are self-explanatory)
│   │   │   ├── abs.scala
│   │   │   ├── add.scala
│   │   │   ├── bias_add.scala
│   │   │   ├── cast.scala
│   │   │   ├── concatenate.scala
│   │   │   ├── div.scala
│   │   │   ├── exp.scala
│   │   │   ├── flatten.scala
│   │   │   ├── im2col.scala       # image -> column transform (conv implem helper)
│   │   │   ├── matmul.scala
│   │   │   ├── mul.scala
│   │   │   ├── reciprocal.scala   # 1/x
│   │   │   ├── repack.scala       # bit repacking
│   │   │   ├── reshape.scala
│   │   │   ├── rsqrt.scala        # 1/sqrt(x)
│   │   │   ├── scale_add.scala
│   │   │   ├── seq2col.scala      # sequence -> column transform (conv implem helper)
│   │   │   ├── slice.scala
│   │   │   ├── sqrt.scala
│   │   │   ├── sub.scala
│   │   │   └── transpose.scala
│   │   ├── activations/
│   │   │   ├── relu.scala
│   │   │   ├── leaky_relu.scala
│   │   │   └── softmax.scala      # built from exp + reciprocal components
│   │   ├── poolings/
│   │   │   ├── avgpool1d.scala
│   │   │   └── maxpool1d.scala
│   │   ├── layers/
│   │   │   ├── Conv1D.scala
│   │   │   ├── Conv2D.scala
│   │   │   ├── Linear.scala       # fully-connected layer
│   │   │   ├── batchnorm.scala
│   │   │   └── layernorm.scala
│   │   ├── memory/
│   │   │   ├── DMAReader.scala        # DMA read over AXI4
│   │   │   ├── DMAReader2D.scala      # 2D DMA read (FSM-based, for image-like data)
│   │   │   └── StreamDoubleBuffer.scala  # double buffer to overlap transfer and compute
│   │   ├── nn/                  # neural-network composition
│   │   │   ├── LayerSpec.scala  # declarative description of an NN layer
│   │   │   ├── Sequential.scala # stack of layers executed sequentially
│   │   │   └── Accelerator.scala# top-level accelerator wrapper (AXI4-based)
│   │   ├── accelerator/
│   │   │   └── MLAccelerator.scala  # main accelerator wiring the whole system together
│   │   ├── examples/            # runnable components, each generates Verilog
│   │   │   ├── Template.scala       # minimal: single op + tensor stream
│   │   │   ├── SimplePipeline.scala # multi-stage pipeline
│   │   │   ├── SimpleCNN.scala      # small CNN built with layers
│   │   │   ├── SequentialCNN.scala  # CNN built with the nn.Sequential API
│   │   │   ├── Comprehensive1DCNN.scala  # larger 1D CNN
│   │   │   ├── HighLevelTemplate.scala   # template for the high-level nn API
│   │   │   └── DMATemplate.scala        # template with AXI4 DMA memory access
│   │   └── utils/
│   │       ├── Float.scala     # float bit-manipulation helpers
│   │       ├── PWL.scala       # piece-wise linear approximation (for transcendentals)
│   │       └── math_luts.scala # look-up tables for math functions
│   └── test/src/spinalML/      # ScalaTest mirrors of src/, one *Test.scala per file
│       ├── dtypes/             # FP4Test, FP8Test, FloatMathTest, I8Test
│       ├── activations/        # ReLUTest, LeakyReLUTest, SoftmaxTest
│       ├── examples/           # TemplateTest, SimpleCNNTest, SimplePipelineTest,
│       │                       #   HighLevelTemplateTest, DMATemplateTest
│       ├── interfaces/         # TensorToAxi4StreamTest, Axi4StreamToTensorTest
│       ├── layers/             # Conv1DTest, Conv2DTest, LinearTest, BatchNormTest, LayerNormTest
│       ├── memory/             # DMAReaderTest, DMAReader2DTest, StreamDoubleBufferTest,
│       │                       #   AxiArbiterStressTest
│       ├── ops/                # one *Test.scala per file in src/ops
│       ├── symbolicTest/       # formal (symbolic) specs, mirroring src/ packages
│       │   └── ops/AddFormal.scala  # proof of the I8 add op (k-induction, Z3)
│       ├── poolings/           # AvgPool1DTest, MaxPool1DTest
│       ├── accelerator/        # MLAcceleratorTest
│       ├── tensors/            # TensorSim.scala (simulation driver)
│       └── test/               # SequentialTest, SequentialCNNTest, Comprehensive1DCNNTest
└── tests/
    └── python/
        ├── conftest.py         # pytest options (--debug-math -> true_math_errors.log)
        ├── golden_models/      # NumPy bit-exact reference of the hardware
        │   ├── dtypes.py       # FloatML format in Python
        │   └── ops.py          # golden model of each HW operation
        ├── utils/              # cocotb testbench helpers
        │   ├── tb_utils.py         # runs Mill to generate Verilog into sim_build/, copies ROMs
        │   ├── cocotb_helpers.py   # generic unary-op test runner (bit-exact checks)
        │   ├── softmax_helper.py   # softmax (lanes=4) test runner
        │   └── test_layers_utils.py# layer-level test helpers
        └── test_*.py           # one per op/layer: test_add, test_mul, test_conv2d,
                                #   test_softmax, test_dtypes, test_reshape_flatten, ...
```

## Conventions

- **Scala tests** mirror the source: `spinalML/test/src/spinalML/<pkg>/XxxTest.scala` tests `spinalML/src/spinalML/<pkg>/xxx.scala`.
- **Symbolic tests** live in `spinalML/test/src/spinalML/symbolicTest/<pkg>/XxxFormal.scala` (one formal proof per component, discovered by the `*Formal.scala` glob in `ci-symbolic.yml`).
- **Python tests** pair each operation/layer with a NumPy golden model in `golden_models/`, verified bit-exactly through Cocotb + Verilator.